import { createComputeRoute } from "@/lib/custom-compute-route";
import { createTrainingHandler } from "@/lib/custom-validation-service";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 180;
export const POST = createComputeRoute(createTrainingHandler({ retainModels: true }), "train");
