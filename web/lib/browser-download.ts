import type { DownloadArtifact } from "./custom-result-export.ts";
export function downloadArtifact(artifact:DownloadArtifact):void {
  const url=URL.createObjectURL(new Blob([artifact.content],{type:artifact.mime}));
  const link=document.createElement("a");link.href=url;link.download=artifact.filename;
  try{document.body.append(link);link.click();}finally{link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
}
