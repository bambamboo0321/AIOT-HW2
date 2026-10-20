// Read-only public candidate audit. Never opens .env, credential containers or ignored caches.
import {execFileSync} from 'node:child_process';
import {readFile,lstat} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const git=(...args)=>execFileSync('git',args,{cwd:root}).toString().trim();
const lines=value=>value?value.split('\n'):[];
const modified=lines(git('diff','--name-only'));
const untracked=lines(git('ls-files','--others','--exclude-standard'));
const allowed=p=>p.startsWith('web/')||['.gitignore','README.md','docs/vercel_deployment_plan.md'].includes(p);
const files=[...new Set([...modified,...untracked].filter(allowed))].sort();
const findings=[],sizes=[];
const forbidden=/(^|\/)(\.env(?:\..*)?|node_modules|\.next|\.vercel|\.venv|\.native-env|\.npm-cache|artifacts|coverage|__pycache__)(\/|$)|\.(?:pem|key|p12|pfx|pyc|wasm|whl|joblib)$|BostonHousing\.csv$/;
const rules=[['private-key',/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/],['github-token',/\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b/],['aws-access-id',/\bAKIA[A-Z0-9]{16}\b/],['private-local-path',/\/Users\/[^/\s]+\/|\/home\/[^/\s]+\//]];
for(const path of files){
 if(forbidden.test(path)){findings.push({path,issue:'forbidden candidate'});continue;}
 const stat=await lstat(resolve(root,path));if(!stat.isFile()){findings.push({path,issue:'non-regular file'});continue;}
 const bytes=await readFile(resolve(root,path));sizes.push({path,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
 if(bytes.length>2*1024*1024)findings.push({path,issue:'large candidate >2 MiB'});
 for(const [issue,pattern] of rules)if(pattern.test(bytes.toString('utf8')))findings.push({path,issue});
}
for(const p of modified.filter(p=>!allowed(p)))findings.push({path:p,issue:'unexpected tracked modification'});
for(const path of ['web/.npmrc','backend/.env.example']){
 const content=await readFile(resolve(root,path),'utf8');
 if(/_authToken\s*=|_password\s*=|-----BEGIN .*PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}\b/.test(content))findings.push({path,issue:'credential configuration requires review'});
}
const protectedDiff=lines(git('diff','61478fc69f425d2928d6d5444abdb95de6f8035e','--name-only','--','src','backend','reports','data','dashboard_data','web/app/api','web/lib/custom-compute-proxy.ts','web/lib/custom-validation-service.ts','web/lib/custom-model-store.ts'));
const output={branch:git('branch','--show-current'),head:git('rev-parse','HEAD'),staging_empty:!git('diff','--cached','--name-only'),candidate_count:files.length,candidate_bytes:sizes.reduce((n,p)=>n+p.bytes,0),new_files:files.filter(p=>untracked.includes(p)).length,candidates:sizes,excluded_untracked_roots:[...new Set(untracked.filter(p=>!allowed(p)).map(p=>p.split('/')[0]))],protected_changes:protectedDiff,findings,scope:'changed/new M17 release files; previously published M16 artifacts are not restaged; pattern scan is not an absolute secret guarantee'};
console.log(JSON.stringify(output,null,2));
if(findings.length||protectedDiff.length||!output.staging_empty)process.exitCode=1;
