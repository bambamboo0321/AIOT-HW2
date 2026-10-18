import Link from "next/link";
export default function NotFound() { return <div className="page-container"><div className="coming-card"><span className="outline-tag">404</span><h1>找不到這個研究頁面</h1><p>請由導覽選擇頁面，或返回 Overview。</p><Link className="primary-link" href="/">返回 Overview</Link></div></div>; }
