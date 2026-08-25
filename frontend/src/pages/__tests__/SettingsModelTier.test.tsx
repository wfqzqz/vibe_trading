import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Settings } from "../Settings";

const apiMock = vi.hoisted(() => ({
  getLLMSettings: vi.fn(),
  getDataSourceSettings: vi.fn(),
  getChannelStatus: vi.fn(),
  listLLMModels: vi.fn(),
  startChannels: vi.fn(),
  stopChannels: vi.fn(),
  updateLLMSettings: vi.fn(),
  updateDataSourceSettings: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: apiMock,
    isAuthRequiredError: vi.fn(() => false),
  };
});

vi.mock("@/lib/apiAuth", () => ({
  getApiAuthKey: vi.fn(() => ""),
  setApiAuthKey: vi.fn(),
}));

function deepseekSettings(modelName = "deepseek-v4-pro", tier = "pro") {
  return {
    provider: "deepseek",
    model_name: modelName,
    model_tier: tier,
    base_url: "https://api.deepseek.com/v1",
    api_key_env: "DEEPSEEK_API_KEY",
    api_key_configured: true,
    api_key_required: true,
    temperature: 0.0,
    timeout_seconds: 120,
    max_retries: 2,
    reasoning_effort: "",
    sse_timeout_seconds: 300,
    env_path: "agent/.env",
    providers: [
      {
        name: "deepseek",
        label: "DeepSeek",
        api_key_env: "DEEPSEEK_API_KEY",
        base_url_env: "DEEPSEEK_BASE_URL",
        default_model: "deepseek-v4-pro",
        default_base_url: "https://api.deepseek.com/v1",
        api_key_required: true,
        auth_type: "api_key",
        model_tiers: {
          flash: "deepseek-v4-flash",
          pro: "deepseek-v4-pro",
        },
      },
    ],
  };
}

function dataSourceSettings() {
  return {
    tushare_token_configured: false,
    baostock_supported: true,
    baostock_installed: true,
    baostock_message: "BaoStock available",
    env_path: "agent/.env",
  };
}

function channelStatus() {
  return {
    running: false,
    inbound_queue: 0,
    outbound_queue: 0,
    session_count: 0,
    channels: {
      websocket: {
        name: "websocket",
        display_name: "WebSocket",
        configured: true,
        enabled: true,
        available: true,
        loaded: true,
        running: false,
        error: "",
        install_hint: "",
      },
    },
  };
}

describe("Settings model-tier contract", () => {
  beforeEach(() => {
    // Suppress the local-API-credential form so the LLM form's "Save" button
    // is unambiguous in the test.
    Object.defineProperty(window, "vibeDesktop", {
      configurable: true,
      value: { isDesktop: true },
    });
    apiMock.getLLMSettings.mockResolvedValue(deepseekSettings());
    apiMock.getDataSourceSettings.mockResolvedValue(dataSourceSettings());
    apiMock.getChannelStatus.mockResolvedValue(channelStatus());
    apiMock.listLLMModels.mockResolvedValue({
      provider: "deepseek",
      models: ["deepseek-v4-pro", "deepseek-v4-flash"],
      source: "provider",
    });
    apiMock.updateLLMSettings.mockImplementation(async (payload) => {
      return deepseekSettings(payload.model_name, payload.model_tier);
    });
  });

  it("renders the tier selector for a provider that declares model_tiers", async () => {
    render(<Settings />);
    await screen.findByText("LLM Settings");

    expect(screen.getByText("Model Tier")).toBeInTheDocument();
    expect(screen.getByText("flash")).toBeInTheDocument();
    expect(screen.getByText("pro")).toBeInTheDocument();
    expect(screen.getByText("deepseek-v4-flash")).toBeInTheDocument();
    expect(screen.getByText("deepseek-v4-pro")).toBeInTheDocument();
  });

  it("does not render the tier selector for a provider without model_tiers", async () => {
    apiMock.getLLMSettings.mockResolvedValue({
      ...deepseekSettings("claude-sonnet-5", ""),
      provider: "openrouter",
      model_name: "deepseek/deepseek-v4-pro",
      model_tier: "",
      providers: [
        {
          name: "openrouter",
          label: "OpenRouter",
          api_key_env: "OPENROUTER_API_KEY",
          base_url_env: "OPENROUTER_BASE_URL",
          default_model: "deepseek/deepseek-v4-pro",
          default_base_url: "https://openrouter.ai/api/v1",
          api_key_required: true,
          auth_type: "api_key",
        },
      ],
    });

    render(<Settings />);
    await screen.findByText("LLM Settings");

    expect(screen.queryByText("Model Tier")).not.toBeInTheDocument();
  });

  it("switching tier updates the model and persists model_tier on save", async () => {
    render(<Settings />);
    await screen.findByText("LLM Settings");

    fireEvent.click(screen.getByText("flash"));

    // The form should now point at the flash model.
    await waitFor(() => {
      expect(screen.getByDisplayValue("deepseek-v4-flash")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(apiMock.updateLLMSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          model_tier: "flash",
          model_name: "deepseek-v4-flash",
        }),
      );
    });
  });
});
