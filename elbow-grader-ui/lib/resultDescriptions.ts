import type { GeometricData } from "@/lib/types";

export const BAUMANN_STATUS_MESSAGES: Record<string, string> = {
  no_yolo_ckpt: "YOLO humerus model checkpoint not found on the server. Contact the administrator.",
  no_image: "Baumann angle not available — provide an AP image to enable this.",
  no_mask: "Humerus was not detected in this image — check image quality or field of view.",
  no_shaft: "Could not fit a shaft axis — ensure the full humerus shaft is visible.",
  no_physis: "Could not fit the physeal line — check image quality and distal humerus visibility.",
};

export function getBaumannUnavailableMessage(status?: string | null) {
  if (!status) return "Baumann angle not available — provide an AP image to enable this.";
  if (status === "ok") return null;
  if (BAUMANN_STATUS_MESSAGES[status]) return BAUMANN_STATUS_MESSAGES[status];
  if (status.startsWith("error:")) return `Pipeline error: ${status.replace("error:", "").trim()}`;
  return `Baumann pipeline could not complete (status: ${status}).`;
}

export function getAhlUnavailableMessage(geometric: GeometricData | null) {
  if (!geometric || geometric.skipped_reason) {
    return `Geometric track skipped: ${geometric?.skipped_reason ?? "no LAT input"}`;
  }
  return null;
}

export function getBoneProfileUnavailableMessage(geometric: GeometricData | null) {
  if (!geometric?.width_profile) {
    return "No cortical width data — LAT geometric track did not run or did not reach Grade 2.";
  }
  return null;
}
