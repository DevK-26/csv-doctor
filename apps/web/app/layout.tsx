export const metadata = {
  title: "CSV Doctor — What's wrong, weird and interesting in your CSV?",
  description:
    "Drop any CSV. Real ML profiles it, finds anomalies and drift, and a free AI explains it in plain English. No signup. Files auto-deleted.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0 }}>{children}</body>
    </html>
  );
}
