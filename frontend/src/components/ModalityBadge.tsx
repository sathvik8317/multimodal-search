import { MODALITY_BADGE, MODALITY_LABEL, type Modality } from "../modality";

export function ModalityBadge({ modality }: { modality: Modality }) {
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${MODALITY_BADGE[modality]}`}
    >
      {MODALITY_LABEL[modality]}
    </span>
  );
}
