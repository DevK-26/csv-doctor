export const int = (n: number): string => n.toLocaleString("en-US");
export const dec = (n: number, digits = 3): string => n.toFixed(digits);
export const signed = (n: number): string => (n >= 0 ? `+${n}` : String(n));

/** Compact human number for profile cells (the engine already rounded them). */
export function cell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? int(v) : String(v);
  return String(v);
}
