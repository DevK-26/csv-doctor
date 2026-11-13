// Hands the chosen File from the landing page to the report page. Module state survives
// client-side navigation (router.push) but nothing is persisted: a reload starts clean
// and the file never leaves the browser except in requests to the engine.
let pending: File | null = null;

export function setPendingFile(file: File): void {
  pending = file;
}

export function takePendingFile(): File | null {
  const file = pending;
  pending = null;
  return file;
}
