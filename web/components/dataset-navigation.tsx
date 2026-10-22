"use client";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { DATASET_IDS, DATASET_REGISTRY, DEFAULT_DATASET_ID, type DatasetDefinition } from "@/lib/dataset-registry";
import { datasetForPath, datasetSwitchHref, researchHref, workspaceHref, type WorkspacePath } from "@/lib/dataset-routing";
import { Icon, type IconName } from "./icon";

export type NavigationItem = { href: WorkspacePath; label: string; icon: IconName };
type Props = { items: NavigationItem[]; onNavigate: () => void };
function NavigationContents({ items, onNavigate, pathname, dataset, onSelect }: Props & {
  pathname: string; dataset: DatasetDefinition | null; onSelect?: (value: string) => void;
}) {
  const id = dataset?.id ?? DEFAULT_DATASET_ID;
  return <>
    <Link href={researchHref("/", id)} className="brand" onClick={onNavigate}>
      <span className="brand-mark"><Icon name="energy" size={23}/></span>
      <span><strong>AIOT Research<span className="brand-lab">Lab</span></strong><small>AIOT RESEARCH WORKSPACE</small></span>
    </Link>
    {dataset ? <div className="dataset-picker">
      <label htmlFor="research-dataset">研究資料集</label>
      <select id="research-dataset" aria-describedby="research-dataset-status" value={id} disabled={!onSelect} onChange={event => onSelect?.(event.target.value)}>
        {DATASET_IDS.map(key => <option key={key} value={key}>{DATASET_REGISTRY[key].shortName}</option>)}
      </select>
      <p id="research-dataset-status">{dataset.features.researchPages === "connected" ? "既有研究成果" : (pathname === "/" && dataset.features.overview === "connected") || (pathname === "/feature-selection" && dataset.features.featureSelection === "connected") || (pathname === "/model-evaluation" && dataset.features.modelEvaluation === "connected") ? "Overview／選欄／模型評估已接入" : "此研究頁尚未接入 · 其他研究頁可瀏覽"}</p>
    </div> : <p className="dataset-independent">自訂 CSV · 獨立工作區</p>}
    <div className="nav-section-label">WORKSPACE</div>
    <nav>{items.map(item => <Link key={item.href} href={workspaceHref(item.href, id)} prefetch={false}
      className={`nav-link ${pathname === item.href ? "active" : ""}`} aria-current={pathname === item.href ? "page" : undefined} onClick={onNavigate}>
      <Icon name={item.icon}/><span>{item.label}</span>
    </Link>)}</nav>
  </>;
}
export function DatasetNavigation(props: Props) {
  const pathname = usePathname();
  const query = useSearchParams();
  const router = useRouter();
  const dataset = datasetForPath(pathname, query);
  function select(value: string) {
    const href = datasetSwitchHref(pathname, query.toString(), value, window.location.hash);
    if (href) router.push(href, { scroll: false });
  }
  return <NavigationContents {...props} pathname={pathname} dataset={dataset} onSelect={select}/>;
}
export function DatasetNavigationFallback(props: Props & { pathname: string }) {
  const dataset = props.pathname === "/custom-dataset" ? null : DATASET_REGISTRY[DEFAULT_DATASET_ID];
  return <NavigationContents {...props} dataset={dataset}/>;
}
