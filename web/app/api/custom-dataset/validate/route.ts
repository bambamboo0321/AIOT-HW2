import { createComputeRoute } from "@/lib/custom-compute-route";
import { createValidationHandler } from "@/lib/custom-validation-service";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 180;
export const POST = createComputeRoute(createValidationHandler(), "validate");
