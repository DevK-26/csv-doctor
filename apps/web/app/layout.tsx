import "./globals.css";

export const metadata = {
  title: "CSV Doctor — What's wrong, weird and interesting in your CSV?",
  description:
    "Drop any CSV. Real ML profiles it, finds anomalies and drift, and finds duplicates. Every number is computed, and answers about your data are verified before they are shown.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
