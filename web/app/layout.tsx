import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Overview | Appliances Energy Prediction Lab", template: "%s | AIOT Research Lab" },
  description: "Machine Learning · Feature Selection · Model Evaluation. A traceable AIOT research workspace.",
  robots: { index: false, follow: false },
};

const themeScript = `try{const saved=localStorage.getItem('aep-theme');document.documentElement.dataset.theme=(saved==='dark'||saved==='light')?saved:(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}catch{document.documentElement.dataset.theme=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="zh-Hant" suppressHydrationWarning><head><script dangerouslySetInnerHTML={{ __html: themeScript }}/></head>
    <body><Shell>{children}</Shell></body></html>;
}
