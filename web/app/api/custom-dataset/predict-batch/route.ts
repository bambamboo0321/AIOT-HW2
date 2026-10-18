import { createComputeRoute } from "@/lib/custom-compute-route";
import { createPredictionHandler } from "@/lib/custom-prediction-service";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 180;
export const POST = createComputeRoute(createPredictionHandler({batch:true}), "predict-batch");
