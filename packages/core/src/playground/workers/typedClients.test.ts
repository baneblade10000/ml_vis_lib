import { describe, expect, it } from "vitest";
import type { CnnConfig } from "../cnn/engine";
import type { PlaygroundConfig } from "../types";
import { CnnTrainClient } from "./CnnTrainClient";
import { MlpTrainClient } from "./MlpTrainClient";
import { NetworkTrainClient } from "./NetworkTrainClient";
import { TrainWorkerClient } from "./TrainWorkerClient";
import type {
  CnnTrainSnapshot,
  FromTrainWorker,
  MlpTrainSnapshot,
  NetworkTrainSnapshot,
  TrainSnapshot,
} from "./protocol";
import { FakeWorker } from "./test-helpers/FakeWorker";

function asWorker(w: FakeWorker): Worker {
  return w as unknown as Worker;
}

/**
 * Build an INITIALIZED transport backed by a FakeWorker. The fake acks `init`
 * with a ready snapshot; init must complete because TrainWorkerClient.post is a
 * no-op until the worker is set during init (commands/rebuilds would vanish).
 */
async function makeTransport(readySnapshot: TrainSnapshot): Promise<{
  transport: TrainWorkerClient;
  fake: FakeWorker;
}> {
  const fake = new FakeWorker();
  fake.onPost = (msg) => {
    if ((msg as { type: string }).type === "init") {
      fake.emit({ type: "ready", snapshot: readySnapshot } satisfies FromTrainWorker);
    }
  };
  const transport = new TrainWorkerClient({ createWorker: () => asWorker(fake) });
  await transport.init({});
  return { transport, fake };
}

/** Push a worker→main snapshot tick through the transport. */
function emitTick(fake: FakeWorker, snapshot: TrainSnapshot): void {
  fake.emit({ type: "tick", snapshot } satisfies FromTrainWorker);
}

const cnnSnap = { kind: "cnn" } as unknown as CnnTrainSnapshot;
const netSnap = { kind: "network" } as unknown as NetworkTrainSnapshot;
const mlpSnap = { kind: "mlp" } as unknown as MlpTrainSnapshot;

describe("CnnTrainClient", () => {
  it("onTick fires only for cnn snapshots", async () => {
    const { transport, fake } = await makeTransport(cnnSnap);
    const ticks: CnnTrainSnapshot[] = [];
    new CnnTrainClient({ client: transport, onTick: (s) => ticks.push(s) });

    emitTick(fake, netSnap);
    emitTick(fake, mlpSnap);
    expect(ticks).toHaveLength(0);

    emitTick(fake, cnnSnap);
    expect(ticks).toHaveLength(1);
    expect(ticks[0]!.kind).toBe("cnn");
  });

  it("command() delegates name + args to the transport", async () => {
    const { transport, fake } = await makeTransport(cnnSnap);
    const cnn = new CnnTrainClient({ client: transport, onTick: () => {} });
    cnn.command("setDataset", { dataset: "digits" as CnnConfig["dataset"] });
    cnn.command("regenerateData");

    const commands = fake.messages.filter((m) => (m as { type: string }).type === "command");
    expect(commands).toEqual([
      { type: "command", name: "setDataset", args: { dataset: "digits" } },
      { type: "command", name: "regenerateData", args: undefined },
    ]);
  });

  it("rebuild('mode', ...) delegates reason + payload", async () => {
    const { transport, fake } = await makeTransport(cnnSnap);
    const cnn = new CnnTrainClient({ client: transport, onTick: () => {} });
    cnn.rebuild("mode", "1d");
    const rebuilds = fake.messages.filter((m) => (m as { type: string }).type === "rebuild");
    expect(rebuilds).toEqual([{ type: "rebuild", reason: "mode", payload: "1d" }]);
  });
});

describe("NetworkTrainClient", () => {
  it("onTick fires only for network snapshots", async () => {
    const { transport, fake } = await makeTransport(netSnap);
    const ticks: NetworkTrainSnapshot[] = [];
    new NetworkTrainClient({ client: transport, onTick: (s) => ticks.push(s) });

    emitTick(fake, cnnSnap);
    emitTick(fake, mlpSnap);
    expect(ticks).toHaveLength(0);

    emitTick(fake, netSnap);
    expect(ticks).toHaveLength(1);
  });
});

describe("MlpTrainClient", () => {
  it("onTick fires only for mlp snapshots", async () => {
    const { transport, fake } = await makeTransport(mlpSnap);
    const ticks: MlpTrainSnapshot[] = [];
    new MlpTrainClient({ client: transport, onTick: (s) => ticks.push(s) });

    emitTick(fake, cnnSnap);
    emitTick(fake, netSnap);
    expect(ticks).toHaveLength(0);

    emitTick(fake, mlpSnap);
    expect(ticks).toHaveLength(1);
  });

  it("rebuild('reset', config) delegates reason + payload", async () => {
    const { transport, fake } = await makeTransport(mlpSnap);
    const mlp = new MlpTrainClient({ client: transport, onTick: () => {} });
    const cfg = { kind: "cfg" } as unknown as PlaygroundConfig;
    mlp.rebuild("reset", cfg);
    const rebuilds = fake.messages.filter((m) => (m as { type: string }).type === "rebuild");
    expect(rebuilds).toEqual([{ type: "rebuild", reason: "reset", payload: cfg }]);
  });
});
