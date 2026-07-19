import json
import re
from typing import Any, Iterable, Sequence

import ollama

from config import (
    MAX_CODE_FACTS_FOR_LLM,
    MAX_EXTRACTOR_EXCERPT_CHARS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
)
from memory_schema import CodeChunk, EngineeringMemory


EXTRACTION_PROMPT = """
You are an engineering memory extractor.

Extract reusable implementation intelligence from the provided engineering session.

Return ONLY valid JSON with EXACTLY these keys:
- feature
- domain
- language
- framework
- complexity
- implementation_patterns
- architecture_decisions
- implementation_flow
- generated_files
- must_preserve_conventions

Rules:
- Do not include explanations outside JSON.
- Treat deterministic facts in the input as authoritative.
- Do not invent generated files.
- Do not invent a language or framework when deterministic facts already identify them.
- implementation_patterns must describe reusable code or architecture patterns.
- architecture_decisions must describe architectural choices.
- implementation_flow must describe execution/build/data flow.
- generated_files must list only detected or clearly mentioned file paths.
- must_preserve_conventions must list exact conventions such as return types, route formats, response shapes, DI style, middleware order, timestamp style, DTO mapping style, repository/service style, validation style, exception handling style, and persistence style.
- complexity must be one of: low, medium, high.
"""
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "feature": {"type": "string"},
        "domain": {"type": "string"},
        "language": {"type": "string"},
        "framework": {"type": "string"},
        "complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "implementation_patterns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "architecture_decisions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "implementation_flow": {
            "type": "array",
            "items": {"type": "string"},
        },
        "generated_files": {
            "type": "array",
            "items": {"type": "string"},
        },
        "must_preserve_conventions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "feature",
        "domain",
        "language",
        "framework",
        "complexity",
        "implementation_patterns",
        "architecture_decisions",
        "implementation_flow",
        "generated_files",
        "must_preserve_conventions",
    ],
}

def ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        parts = value.splitlines() if "\n" in value else value.split(",")
        return [part.strip(" -\t") for part in parts if part.strip(" -\t")]

    return []


def safe_json_loads(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1

        if start == -1 or end <= start:
            raise

        data = json.loads(content[start:end])

    if not isinstance(data, dict):
        raise ValueError("Extractor response JSON must be an object.")

    return data


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []

    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()

        if not cleaned or key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lower_text = text.lower()
    return any(term.lower() in lower_text for term in terms)


def guess_feature(session_text: str) -> str:
    feature_match = re.search(
        r"(?im)^\s*(?:feature|project|title|topic)\s*:\s*(.+?)\s*$",
        session_text,
    )

    if feature_match:
        return feature_match.group(1).strip()[:120]

    api_match = re.search(
        r"(?i)([A-Za-z0-9][A-Za-z0-9 ._-]{2,80}\s+(?:API|Web API|Management API))",
        session_text,
    )

    if api_match:
        return re.sub(r"\s+", " ", api_match.group(1)).strip()

    if contains_any(
        session_text,
        ["idempotency-key", "idempotency key", "idempotent post"],
    ):
        return "API Idempotency"

    return "unknown implementation"


def guess_language(session_text: str) -> str:
    if re.search(
        r"```(?:csharp|cs)\b|\.cs\b|ASP\.NET|C#|\.NET",
        session_text,
        re.IGNORECASE,
    ):
        return "C#"

    if re.search(
        r"```(?:typescript|ts)\b|\.ts\b|Angular",
        session_text,
        re.IGNORECASE,
    ):
        return "TypeScript"

    if re.search(
        r"```(?:javascript|js)\b|\.js\b|Node\.js",
        session_text,
        re.IGNORECASE,
    ):
        return "JavaScript"

    if re.search(
        r"```(?:python|py)\b|\.py\b|FastAPI",
        session_text,
        re.IGNORECASE,
    ):
        return "Python"

    return "unknown"


def guess_framework(session_text: str) -> str:
    lower_text = session_text.lower()

    has_csharp_web_project = (
        "microsoft.net.sdk.web" in lower_text
        or "microsoft.aspnetcore" in lower_text
        or "builder.services.addcontrollers" in lower_text
        or "app.mapcontrollers" in lower_text
        or "[apicontroller]" in lower_text
    )

    has_net8 = (
        "asp.net core 8" in lower_text
        or "net8.0" in lower_text
        or "<targetframework>net8.0</targetframework>" in lower_text
        or "entity framework core 8" in lower_text
        or "microsoft.entityframeworkcore\" version=\"8" in lower_text
        or "microsoft.entityframeworkcore.sqlserver\" version=\"8" in lower_text
    )

    has_ef_core = (
        "entity framework core" in lower_text
        or "microsoft.entityframeworkcore" in lower_text
        or "dbcontext" in lower_text
        or "dbset<" in lower_text
        or "usesqlserver" in lower_text
        or "adddbcontext<" in lower_text
    )

    if has_csharp_web_project and has_net8:
        return "ASP.NET Core 8"

    if "asp.net core 8" in lower_text:
        return "ASP.NET Core 8"

    if has_csharp_web_project:
        return "ASP.NET Core"

    if has_ef_core and has_net8:
        return "EF Core 8"

    if "asp.net core" in lower_text:
        return "ASP.NET Core"

    if "angular" in lower_text:
        return "Angular"

    if "fastapi" in lower_text:
        return "FastAPI"

    if "react" in lower_text:
        return "React"

    return "unknown"


def guess_domain(session_text: str, feature: str) -> str:
    if contains_any(session_text, ["idempotency-key", "idempotency key", "idempotent"]):
        return "API Design"

    if feature and feature != "unknown implementation":
        domain = re.sub(
            r"\b(API|Web API|Management API)\b",
            "",
            feature,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", domain).strip() or "general"

    return "general"


def extract_file_paths(session_text: str) -> list[str]:
    paths = re.findall(
        r"(?im)(?:^|\s)([\w .{}()/-]+\.(?:cs|csproj|json|yml|yaml|js|ts|py|sql|md)|\.gitignore)",
        session_text,
    )

    cleaned: list[str] = []

    for path in paths:
        normalized = path.strip().replace("\\", "/").strip("`'\" ,")

        if normalized and normalized not in cleaned:
            cleaned.append(normalized)

    return cleaned[:80]


def summarize_code_chunks(code_chunks: Sequence[CodeChunk]) -> list[str]:
    return [
        f"- {chunk.file_path} | language={chunk.language} | chunk_type={chunk.chunk_type} | symbol={chunk.symbol_name}"
        for chunk in code_chunks[:MAX_CODE_FACTS_FOR_LLM]
    ]


def infer_patterns_from_code_chunks(
    code_chunks: Sequence[CodeChunk],
    session_text: str,
) -> list[str]:
    chunk_types = {chunk.chunk_type for chunk in code_chunks}
    patterns: list[str] = []

    if "controller" in chunk_types:
        patterns.append("Controller-based API endpoint pattern.")

    if "service" in chunk_types:
        patterns.append("Service layer pattern for business logic.")

    if "repository" in chunk_types:
        patterns.append("Repository pattern for data access.")

    if "dto" in chunk_types:
        patterns.append("DTO pattern for request and response contracts.")

    if "db_context" in chunk_types:
        patterns.append("EF Core DbContext persistence pattern.")

    if "startup_pipeline" in chunk_types:
        patterns.append("Program.cs startup pipeline configuration pattern.")

    if contains_any(
        session_text,
        ["middleware", "exceptionmiddleware", "globalexception", "usemiddleware"],
    ):
        patterns.append("Global exception handling middleware pattern.")

    if contains_any(session_text, ["fluentvalidation", "validator", "validation"]):
        patterns.append("Request validation pattern.")

    if contains_any(session_text, ["idempotency-key", "idempotency key"]):
        patterns.append("Idempotency-Key header pattern.")

    return dedupe_preserve_order(patterns)


def infer_architecture_decisions(
    session_text: str,
    code_chunks: Sequence[CodeChunk],
) -> list[str]:
    decisions: list[str] = []

    if contains_any(session_text, ["sql server", "usesqlserver", "defaultconnection"]):
        decisions.append(
            "Use SQL Server configured through the DefaultConnection connection string."
        )

    if contains_any(session_text, ["entity framework", "ef core", "dbcontext", "dbset<"]):
        decisions.append("Use EF Core DbContext for persistence.")

    if contains_any(session_text, ["addscoped<", "dependency injection", " di "]):
        decisions.append(
            "Use dependency injection with scoped service and repository registrations."
        )

    if contains_any(session_text, ["swagger", "addswaggergen", "useswaggerui"]):
        decisions.append("Expose API documentation through Swagger/OpenAPI.")

    if contains_any(session_text, ["global exception", "exception middleware", "usemiddleware"]):
        decisions.append(
            "Handle unhandled API errors through centralized exception middleware."
        )

    if contains_any(session_text, ["idempotency-key", "idempotency key"]):
        decisions.append(
            "Use an Idempotency-Key header to make retry-sensitive POST operations safe."
        )

    return dedupe_preserve_order(decisions)


def infer_implementation_flow(
    session_text: str,
    code_chunks: Sequence[CodeChunk],
) -> list[str]:
    chunk_types = {chunk.chunk_type for chunk in code_chunks}
    flow: list[str] = []

    if "controller" in chunk_types and "service" in chunk_types:
        flow.append("Controller receives HTTP request and calls the service layer.")

    if "service" in chunk_types and "repository" in chunk_types:
        flow.append("Service layer applies business logic and calls the repository layer.")

    if "repository" in chunk_types and "db_context" in chunk_types:
        flow.append("Repository layer uses DbContext for database operations.")

    if "dto" in chunk_types:
        flow.append("DTOs are used for request and response payloads.")

    if "startup_pipeline" in chunk_types:
        flow.append("Program.cs configures services, middleware, and controller routing.")

    if contains_any(session_text, ["idempotency-key", "idempotency key"]):
        flow.extend(
            [
                "Client sends an Idempotency-Key with retry-sensitive POST requests.",
                "Server checks the stored key and request hash before processing.",
                "Server returns the stored response for the same key and same request body.",
                "Server rejects the request when the same key is reused with a different body.",
            ]
        )

    return dedupe_preserve_order(flow)


def infer_complexity(session_text: str, code_chunks: Sequence[CodeChunk]) -> str:
    chunk_types = {chunk.chunk_type for chunk in code_chunks}

    if len(code_chunks) >= 12 or len(chunk_types) >= 7:
        return "high"

    if len(code_chunks) >= 4 or len(chunk_types) >= 3:
        return "medium"

    if contains_any(
        session_text,
        ["distributed", "microservice", "idempotency", "authentication", "authorization"],
    ):
        return "medium"

    return "low"


def calculate_quality_score(
    feature: str,
    implementation_patterns: Sequence[str],
    architecture_decisions: Sequence[str],
    implementation_flow: Sequence[str],
    generated_files: Sequence[str],
    must_preserve_conventions: Sequence[str],
    summary: str,
) -> float:
    score = 0.0

    if feature and feature != "unknown implementation":
        score += 1.2

    score += min(len(implementation_patterns) * 0.6, 2.4)
    score += min(len(architecture_decisions) * 0.7, 2.1)
    score += min(len(implementation_flow) * 0.5, 2.0)
    score += min(len(generated_files) * 0.25, 1.5)
    score += min(len(must_preserve_conventions) * 0.45, 2.5)

    summary_length = len(summary.split())

    if summary_length > 80:
        score += 2.0
    elif summary_length > 40:
        score += 1.0

    engineering_terms = [
        "middleware",
        "repository",
        "controller",
        "dto",
        "dbcontext",
        "migration",
        "dependency injection",
        "service layer",
        "repository pattern",
        "authentication",
        "authorization",
        "route",
        "actionresult",
        "notfound",
        "entity framework",
        "createdataction",
        "nocontent",
        "swagger",
        "validation",
        "idempotency",
        "idempotency-key",
    ]

    summary_lower = summary.lower()
    matches = sum(1 for term in engineering_terms if term in summary_lower)

    score += min(matches * 0.25, 2.0)

    return round(score, 2)


def deterministic_memory(
    session_text: str,
    reason: str = "fallback",
    code_chunks: Sequence[CodeChunk] | None = None,
    deterministic_conventions: Sequence[str] | None = None,
) -> EngineeringMemory:
    code_chunks = list(code_chunks or [])
    deterministic_conventions = list(deterministic_conventions or [])

    feature = guess_feature(session_text)
    language = guess_language(session_text)
    framework = guess_framework(session_text)
    domain = guess_domain(session_text, feature)

    generated_files = dedupe_preserve_order(
        extract_file_paths(session_text) + [chunk.file_path for chunk in code_chunks]
    )

    implementation_patterns = infer_patterns_from_code_chunks(code_chunks, session_text)
    architecture_decisions = infer_architecture_decisions(session_text, code_chunks)
    implementation_flow = infer_implementation_flow(session_text, code_chunks)
    conventions = list(deterministic_conventions)

    lower_text = session_text.lower()

    if not implementation_patterns:
        if "repository" in lower_text:
            implementation_patterns.append(
                "Repository pattern with interface and implementation."
            )

        if "service" in lower_text:
            implementation_patterns.append(
                "Service layer pattern with interface and implementation."
            )

        if "dto" in lower_text:
            implementation_patterns.append(
                "DTO pattern for request and response contracts."
            )

        if "controller" in lower_text:
            implementation_patterns.append("Controller-based API endpoints.")

        if (
            "dbcontext" in lower_text
            or "entity framework" in lower_text
            or "ef core" in lower_text
        ):
            implementation_patterns.append("EF Core DbContext data access pattern.")

    if "dto" in lower_text:
        conventions.append(
            "DTOs are separated from entity models for request and response contracts."
        )

    if "addscoped<" in lower_text:
        conventions.append(
            "Dependency injection uses AddScoped<TInterface, TImplementation>() registrations."
        )

    if "ActionResult<" in session_text:
        conventions.append(
            "Controllers use ActionResult<T> return types for typed API responses."
        )

    if "{id:int}" in session_text:
        conventions.append(
            "Controller routes use explicit integer route constraints like {id:int}."
        )

    if "CreatedAtAction" in session_text:
        conventions.append("POST endpoints return CreatedAtAction for created resources.")

    if "NoContent()" in session_text:
        conventions.append(
            "PUT and DELETE endpoints return NoContent() on successful completion."
        )

    if "DateTime.UtcNow" in session_text:
        conventions.append("Entity timestamp fields use DateTime.UtcNow defaults.")

    if contains_any(session_text, ["idempotency-key", "idempotency key"]):
        conventions.extend(
            [
                "Use an Idempotency-Key header for retry-safe POST operations.",
                "Store the request body hash with the idempotency key.",
                "Return the stored response when the same key is reused with the same request body.",
                "Reject the request when the same key is reused with a different request body.",
                "Set a TTL for stored idempotency keys.",
            ]
        )

    complexity = infer_complexity(session_text, code_chunks)
    conventions = dedupe_preserve_order(conventions)

    quality_score = calculate_quality_score(
        feature=feature,
        implementation_patterns=implementation_patterns,
        architecture_decisions=architecture_decisions,
        implementation_flow=implementation_flow,
        generated_files=generated_files,
        must_preserve_conventions=conventions,
        summary=session_text,
    )

    return EngineeringMemory(
        feature=feature,
        domain=domain,
        language=language,
        framework=framework,
        complexity=complexity,
        implementation_patterns=dedupe_preserve_order(implementation_patterns),
        architecture_decisions=dedupe_preserve_order(architecture_decisions),
        implementation_flow=dedupe_preserve_order(implementation_flow),
        generated_files=generated_files,
        summary=session_text,
        quality_score=quality_score,
        source=f"deterministic-{reason}",
        must_preserve_conventions=conventions,
    )


def build_compact_extraction_input(
    session_text: str,
    code_chunks: Sequence[CodeChunk] | None = None,
    deterministic_conventions: Sequence[str] | None = None,
) -> str:
    code_chunks = list(code_chunks or [])
    deterministic_conventions = list(deterministic_conventions or [])

    file_paths = dedupe_preserve_order(
        extract_file_paths(session_text) + [chunk.file_path for chunk in code_chunks]
    )

    deterministic_summary = deterministic_memory(
        session_text=session_text,
        reason="pre-llm-facts",
        code_chunks=code_chunks,
        deterministic_conventions=deterministic_conventions,
    )

    file_text = "\n".join(f"- {path}" for path in file_paths) or "- None detected"
    code_fact_text = "\n".join(summarize_code_chunks(code_chunks)) or "- No code chunks detected"
    convention_text = (
        "\n".join(f"- {item}" for item in deterministic_conventions)
        or "- None detected"
    )

    excerpt = session_text[:MAX_EXTRACTOR_EXCERPT_CHARS]

    return f"""
You are extracting memory from an engineering session.

Use these deterministic facts as authoritative evidence. Do not invent files or technologies that are not supported here.

Deterministic metadata guess:
- feature: {deterministic_summary.feature}
- domain: {deterministic_summary.domain}
- language: {deterministic_summary.language}
- framework: {deterministic_summary.framework}
- complexity: {deterministic_summary.complexity}

Detected code chunks:
{code_fact_text}

Detected file paths:
{file_text}

Deterministically detected conventions:
{convention_text}

Short engineering session excerpt:
{excerpt}
""".strip()


def call_ollama(compact_input: str) -> dict[str, Any]:
    client = ollama.Client(
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )

    return client.chat(
        model=OLLAMA_MODEL,
        format=EXTRACTION_JSON_SCHEMA,
        options={
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_PROMPT,
            },
            {
                "role": "user",
                "content": compact_input,
            },
        ],
    )

def prefer_deterministic_metadata(
    llm_value: Any,
    deterministic_value: str,
    unknown_values: set[str] | None = None,
) -> str:
    unknown_values = unknown_values or {
        "",
        "unknown",
        "unknown implementation",
        "n/a",
        "none",
        "null",
    }

    deterministic_text = str(deterministic_value or "").strip()
    llm_text = str(llm_value or "").strip()

    if deterministic_text and deterministic_text.lower() not in unknown_values:
        return deterministic_text

    if llm_text and llm_text.lower() not in unknown_values:
        return llm_text

    return deterministic_text or llm_text or "unknown"
def normalize_llm_memory(
    data: dict[str, Any],
    session_text: str,
    code_chunks: Sequence[CodeChunk],
    deterministic_conventions: Sequence[str],
) -> EngineeringMemory:
    fallback = deterministic_memory(
        session_text=session_text,
        reason="llm-normalization",
        code_chunks=code_chunks,
        deterministic_conventions=deterministic_conventions,
    )

    implementation_patterns = dedupe_preserve_order(
        ensure_list(data.get("implementation_patterns", []))
        + fallback.implementation_patterns
    )

    architecture_decisions = dedupe_preserve_order(
        ensure_list(data.get("architecture_decisions", []))
        + fallback.architecture_decisions
    )

    implementation_flow = dedupe_preserve_order(
        ensure_list(data.get("implementation_flow", []))
        + fallback.implementation_flow
    )

    generated_files = dedupe_preserve_order(
        ensure_list(data.get("generated_files", []))
        + fallback.generated_files
    )

    must_preserve_conventions = dedupe_preserve_order(
        ensure_list(data.get("must_preserve_conventions", []))
        + fallback.must_preserve_conventions
    )

    feature = str(data.get("feature") or fallback.feature).strip() or fallback.feature
    domain = str(data.get("domain") or fallback.domain).strip() or fallback.domain

    # Deterministic detection is more reliable for language/framework because it is
    # based on file extensions, project files, and concrete code tokens.
    language = prefer_deterministic_metadata(
        llm_value=data.get("language"),
        deterministic_value=fallback.language,
    )

    framework = prefer_deterministic_metadata(
        llm_value=data.get("framework"),
        deterministic_value=fallback.framework,
    )

    complexity = str(data.get("complexity") or fallback.complexity).strip().lower()

    if feature.lower() in {"unknown", "unknown implementation", "n/a", "none", "null"}:
        feature = fallback.feature

    if domain.lower() in {"unknown", "unknown implementation", "n/a", "none", "null"}:
        domain = fallback.domain

    if complexity not in {"low", "medium", "high"}:
        complexity = fallback.complexity

    quality_score = calculate_quality_score(
        feature=feature,
        implementation_patterns=implementation_patterns,
        architecture_decisions=architecture_decisions,
        implementation_flow=implementation_flow,
        generated_files=generated_files,
        must_preserve_conventions=must_preserve_conventions,
        summary=session_text,
    )

    return EngineeringMemory(
        feature=feature,
        domain=domain,
        language=language,
        framework=framework,
        complexity=complexity,
        implementation_patterns=implementation_patterns,
        architecture_decisions=architecture_decisions,
        implementation_flow=implementation_flow,
        generated_files=generated_files,
        summary=session_text,
        quality_score=max(quality_score, fallback.quality_score),
        source=f"ollama-{OLLAMA_MODEL}",
        must_preserve_conventions=must_preserve_conventions,
    )


def extract_engineering_memory(
    session_text: str,
    code_chunks: Sequence[CodeChunk] | None = None,
    deterministic_conventions: Sequence[str] | None = None,
) -> EngineeringMemory:
    code_chunks = list(code_chunks or [])
    deterministic_conventions = list(deterministic_conventions or [])

    compact_input = build_compact_extraction_input(
        session_text=session_text,
        code_chunks=code_chunks,
        deterministic_conventions=deterministic_conventions,
    )

    try:
        response = call_ollama(compact_input)
        content = response["message"]["content"]
        print(content)
        data = safe_json_loads(content)

    except Exception as error:
        print(
            "Extractor failed or timed out. "
            f"Using strengthened deterministic fallback. Error: {error}"
        )
        return deterministic_memory(
            session_text=session_text,
            reason="extractor-error-or-timeout",
            code_chunks=code_chunks,
            deterministic_conventions=deterministic_conventions,
        )

    return normalize_llm_memory(
        data=data,
        session_text=session_text,
        code_chunks=code_chunks,
        deterministic_conventions=deterministic_conventions,
    )