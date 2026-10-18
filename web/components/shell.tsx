"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Icon, type IconName } from "./icon";

const navigation: { href: string; label: string; icon: IconName }[] = [
  { href: "/", label: "Overview", icon: "overview" },
  { href: "/feature-selection", label: "Feature Selection", icon: "features" },
  { href: "/model-evaluation", label: "Model Evaluation", icon: "evaluation" },
  { href: "/prediction-explorer", label: "Prediction Explorer", icon: "prediction" },
  { href: "/custom-dataset", label: "Custom Dataset", icon: "dataset" },
];

function ThemeToggle() {
  function toggle() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("aep-theme", next); } catch { /* Theme still works when storage is unavailable. */ }
  }
  return <button className="icon-button theme-button" aria-label="切換深淺色模式" title="切換深淺色模式" onClick={toggle}>
    <span className="theme-sun"><Icon name="sun" size={19}/></span><span className="theme-moon"><Icon name="moon" size={19}/></span>
  </button>;
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const current = navigation.find(item => item.href === pathname)?.label ?? "Research workspace";
  useEffect(() => {
    function onEscape(event: KeyboardEvent) { if (event.key === "Escape") setOpen(false); }
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, []);
  return <div className="app-shell">
    <a href="#main-content" className="skip-link">跳至主要內容</a>
    {open && <button className="nav-backdrop" aria-label="關閉導覽選單" onClick={() => setOpen(false)}/>}
    <aside id="sidebar" className={`sidebar ${open ? "is-open" : ""}`} aria-label="Research navigation">
      <Link href="/" className="brand" onClick={() => setOpen(false)}>
        <span className="brand-mark"><Icon name="energy" size={23}/></span>
        <span><strong>Energy Prediction<span className="brand-lab">Lab</span></strong><small>AIOT RESEARCH WORKSPACE</small></span>
      </Link>
      <div className="nav-section-label">WORKSPACE</div>
      <nav>{navigation.map(item => <Link key={item.href} href={item.href} prefetch={false}
        className={`nav-link ${pathname === item.href ? "active" : ""}`}
        aria-current={pathname === item.href ? "page" : undefined} onClick={() => setOpen(false)}>
        <Icon name={item.icon}/><span>{item.label}</span>
      </Link>)}</nav>
      <div className="sidebar-note"><span className="tiny-label">THE RESEARCH APPROACH</span>
        <Icon name="book" size={22}/><strong>Evidence before conclusions.</strong>
        <p>以可追溯資料，理解模型的能力與限制。</p><span className="outline-tag">CRISP-DM framework</span>
      </div>
      <div className="sidebar-bottom"><span className="status-dot"/><span>Local workspace</span><span className="version">M14</span></div>
    </aside>
    <div className="workspace">
      <header className="topbar">
        <div className="breadcrumb"><button className="icon-button mobile-menu" aria-label="開啟導覽選單" aria-expanded={open} aria-controls="sidebar" onClick={() => setOpen(!open)}><Icon name={open ? "close" : "menu"}/></button>
          <span className="breadcrumb-root">Workspace</span><span className="slash">/</span><strong>{current}</strong></div>
        <div className="header-actions"><span className="milestone-badge">MILESTONE <b>M14</b></span><span className="header-divider"/><ThemeToggle/></div>
      </header>
      <main id="main-content" tabIndex={-1}>{children}</main>
      <footer className="site-footer"><span>Appliances Energy Prediction Lab</span><span>Research in progress <span className="footer-dot">·</span> Local development</span></footer>
    </div>
  </div>;
}
