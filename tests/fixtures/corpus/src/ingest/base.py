"""Golden fixture code file for the eval/ingest test corpus."""

import time


class PdfIngester:
    def rasterize(self, page):
        """Render a PDF page to a raster image for embedding."""
        return page.get_pixmap()


def _embed_and_write(rows, embedding_client, table):
    """Embed rows and upsert them into the index, retrying transient failures.

    The retry backoff is exponential with base 2, starting at 100ms.
    """
    delay = 0.1
    for attempt in range(3):
        try:
            vectors = embedding_client.embed_documents(rows)
            break
        except ConnectionError:
            time.sleep(delay)
            delay *= 2
    else:
        raise RuntimeError("embedding failed after retries")
    return vectors
