import { describe, expect, it } from "vitest";
import { MonteCarloPiEngine, estimatePi, isInsideUnitQuarterCircle } from "./engine";

describe("monte carlo pi", () => {
  it("detects quarter-circle membership", () => {
    expect(isInsideUnitQuarterCircle(0, 0)).toBe(true);
    expect(isInsideUnitQuarterCircle(1, 0)).toBe(true);
    expect(isInsideUnitQuarterCircle(0.7, 0.7)).toBe(true);
    expect(isInsideUnitQuarterCircle(0.8, 0.8)).toBe(false);
  });

  it("estimates pi from counts", () => {
    expect(estimatePi(785, 1000)).toBeCloseTo(Math.PI, 1);
  });

  it("converges with many samples for a fixed seed", () => {
    const engine = new MonteCarloPiEngine({ seed: 7, batchSize: 500 });
    for (let i = 0; i < 200; i++) engine.addBatch();
    expect(engine.totalSamples).toBe(100_000);
    expect(engine.piEstimate).toBeCloseTo(Math.PI, 2);
  });
});
