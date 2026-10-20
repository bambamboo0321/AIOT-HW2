import {estimatedFits,isAutoMethod} from './custom-selection-policy.ts';
// Same aggregate bound as the existing comparison UI and Python settings().
// This estimate applies only to comparison, never to dataset-wide admission.
export function comparisonWork(methods:readonly string[],ks:readonly number[],features:number,trainingRows:readonly number[]):number{
 const fits=methods.reduce((sum,m)=>sum+(isAutoMethod(m)?[null]:ks).reduce<number>((n,k)=>n+estimatedFits(m,k,features),0),0);
 return fits*features*trainingRows.reduce((sum,n)=>sum+n,0);
}
