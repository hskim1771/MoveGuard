import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const host = (await headers()).get("host") ?? "localhost:3000";
  const protocol = host.includes("localhost") ? "http" : "https";
  const metadataBase = new URL(`${protocol}://${host}`);
  const title = "MoveGuard 작업 자세 정밀평가 MVP";
  const description =
    "실시간 촬영과 영상 분석으로 RULA 결과와 근골격계 유해요인조사 법정서식을 만드는 MoveGuard MVP입니다.";

  return {
    metadataBase,
    title,
    description,
    openGraph: {
      title,
      description,
      type: "website",
      images: [
        {
          url: "/og-figma-ui.png",
          width: 1672,
          height: 941,
          alt: "MoveGuard 작업 자세 정밀평가 MVP",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: ["/og-figma-ui.png"],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
