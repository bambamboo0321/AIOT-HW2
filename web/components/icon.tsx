export type IconName = "energy" | "overview" | "features" | "evaluation" | "prediction" | "dataset" | "sun" | "moon" | "menu" | "close" | "arrow" | "check" | "info" | "book";

const paths: Record<IconName, React.ReactNode> = {
  energy: <path d="m13 2-9 12h7l-1 8 10-13h-7l1-7Z" />,
  overview: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
  features: <><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="8" cy="18" r="2"/></>,
  evaluation: <><path d="M4 4v16h17M8 15l4-5 4 2 5-7"/><path d="M17 5h4v4"/></>,
  prediction: <><path d="M3 17 7 9l5 6 5-10 4 6M3 21h18"/><path d="M3 4h3m3 0h3m3 0h3m3 0h1"/></>,
  dataset: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/></>,
  sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></>,
  moon: <path d="M20.5 14A9 9 0 0 1 10 3.5 9 9 0 1 0 20.5 14Z"/>,
  menu: <path d="M4 6h16M4 12h16M4 18h16"/>, close: <path d="m6 6 12 12M6 18 18 6"/>,
  arrow: <path d="M5 12h14m-5-5 5 5-5 5"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/></>,
  book: <><path d="M12 5v16M3 4c4-1 6 0 9 2 3-2 5-3 9-2v15c-4-1-6 0-9 2-3-2-5-3-9-2V4Z"/></>,
};

export function Icon({ name, size = 20, className = "" }: { name: IconName; size?: number; className?: string }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}>{paths[name]}</svg>;
}
