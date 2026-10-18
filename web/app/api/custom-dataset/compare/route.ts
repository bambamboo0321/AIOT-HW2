import { createComputeRoute } from "@/lib/custom-compute-route";
import { createComparisonHandler } from "@/lib/custom-validation-service";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 180;
export const POST = createComputeRoute(createComparisonHandler(), "compare");
