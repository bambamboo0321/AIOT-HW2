import { randomBytes } from "node:crypto";
import { chmod, copyFile, lstat, mkdtemp, rm } from "node:fs/promises";
import { constants } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { PredictionModel } from "./custom-prediction-types.ts";
type Entry = { info: PredictionModel; path: string; directory: string; leases: number; deleted: boolean; timer: ReturnType<typeof setTimeout> };
type Metadata = Omit<PredictionModel,"id"|"expires_at">;
export class ModelStore {
  private entries = new Map<string,Entry>();
  private root: string; private ttl: number; private capacity: number; private now: ()=>number;
  constructor(options: {root?:string;ttlMs?:number;capacity?:number;now?:()=>number}={}) { this.root=options.root??tmpdir();this.ttl=options.ttlMs??30*60_000;this.capacity=options.capacity??3;this.now=options.now??Date.now; }
  async register(source: string, metadata: Metadata): Promise<PredictionModel> {
    const stat=await lstat(source);if(!stat.isFile()||stat.size===0||stat.size>16*1024*1024)throw new Error("Invalid model file.");
    const directory=await mkdtemp(join(this.root,"aep-model-")); const path=join(directory,"model.joblib");
    try {
      await copyFile(source,path,constants.COPYFILE_EXCL);await chmod(path,0o600);
      const info:PredictionModel={...metadata,features:[...metadata.features],id:randomBytes(32).toString("hex"),expires_at:this.now()+this.ttl};
      // No await while adding/evicting: simultaneous registrations obey capacity.
      while(this.entries.size>=this.capacity) { const id=this.entries.keys().next().value as string; void this.delete(id).catch(()=>{}); }
      const timer=setTimeout(()=>{void this.delete(info.id).catch(()=>{});},this.ttl);timer.unref();
      this.entries.set(info.id,{info,path,directory,leases:0,deleted:false,timer});return structuredClone(info);
    } catch(error) {await rm(directory,{recursive:true,force:true});throw error;}
  }
  async acquire(id:string): Promise<{info:PredictionModel;path:string;release:()=>Promise<void>}|null> {
    const entry=this.entries.get(id);if(!entry)return null;
    if(entry.info.expires_at<=this.now()){await this.delete(id);return null;}
    entry.leases++;let released=false;
    return {info:structuredClone(entry.info),path:entry.path,release:async()=>{if(released)return;released=true;entry.leases--;if(entry.deleted&&!entry.leases)await rm(entry.directory,{recursive:true,force:true});}};
  }
  async delete(id:string):Promise<void> {const entry=this.entries.get(id);if(!entry)return;this.entries.delete(id);entry.deleted=true;clearTimeout(entry.timer);if(!entry.leases)await rm(entry.directory,{recursive:true,force:true});}
  async dispose():Promise<void>{await Promise.all([...this.entries.keys()].map(id=>this.delete(id)));}
}
const shared=globalThis as typeof globalThis & {aepCustomModelStore?:ModelStore};
export function modelStore():ModelStore {return shared.aepCustomModelStore??=new ModelStore();}
