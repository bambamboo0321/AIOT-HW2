import {writeFile} from "node:fs/promises";
import {verifyAllModels} from "../tests/fixed-models-reference.ts";
const report=await verifyAllModels();
await writeFile(new URL("../verification/multi-dataset-phase3fb-numerical.json",import.meta.url),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify(report,null,2));
