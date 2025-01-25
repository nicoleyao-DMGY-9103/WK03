import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const client = await Client.connect("IDMNYU/9103D-2025S-api-example");

Object.defineProperty(window, "predict", {
  get() { return client.predict; },
});
