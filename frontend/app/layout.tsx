import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "健康 Agent",
  description: "个人健康管理 AI 助手",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
