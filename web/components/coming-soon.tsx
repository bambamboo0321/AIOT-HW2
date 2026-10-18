import Link from "next/link";
import { Icon, type IconName } from "./icon";

export function ComingSoon({ title, icon, description }: { title: string; icon: IconName; description: string }) {
  return <div className="page-container coming-page"><div className="eyebrow">RESEARCH WORKSPACE</div>
    <h1>{title}</h1><p className="page-description">{description}</p>
    <section className="coming-card"><span className="coming-icon"><Icon name={icon} size={34}/></span>
      <span className="outline-tag">COMING SOON</span><h2>A space for the next question.</h2>
      <p>此功能尚未實作。M1 已完成 Overview，這裡暫不提供圖表、資料上傳或模型訓練。</p>
      <Link href="/" className="primary-link">返回 Research Overview <Icon name="arrow" size={17}/></Link>
    </section>
  </div>;
}
