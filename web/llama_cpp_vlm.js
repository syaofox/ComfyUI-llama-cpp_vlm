import { app } from "/scripts/app.js";

const REFRESH_URL = "/llama_cpp_vlm/refresh_presets";

async function refreshPresets(node) {
    const combo = node.widgets?.find((w) => w.name === "preset_prompt" && w.type === "combo");
    if (!combo) return;
    try {
        const resp = await fetch(REFRESH_URL, { method: "POST" });
        const data = await resp.json().catch(() => null);
        if (!resp.ok || !data || data.status !== "ok") {
            alert("[llama-cpp_vlm] 刷新预设失败：" + (data?.message || `HTTP ${resp.status}`));
            return;
        }
        const values = data.presets || [];
        if (combo.options && Array.isArray(combo.options)) {
            combo.options = values;
        } else if (combo.options) {
            combo.options.values = values;
        }
        if (!values.includes(combo.value)) {
            combo.value = values.length > 0 ? values[0] : "";
        }
        node.setDirtyCanvas(true, true);
    } catch (e) {
        alert("[llama-cpp_vlm] 刷新预设失败：" + e);
    }
}

app.registerExtension({
    name: "ComfyUI.llama_cpp_vlm.preset_refresh",

    nodeCreated(node) {
        if (node?.comfyClass !== "llama_cpp_instruct_adv") return;

        const btn = node.addWidget("button", "刷新预设", null, () => refreshPresets(node));

        const moveBtnToFront = () => {
            const combo = node.widgets?.find((w) => w.name === "preset_prompt" && w.type === "combo");
            if (!combo) return;
            const btnIdx = node.widgets.indexOf(btn);
            const comboIdx = node.widgets.indexOf(combo);
            if (btnIdx === -1 || comboIdx === -1 || btnIdx < comboIdx) return;
            node.widgets.splice(btnIdx, 1);
            node.widgets.splice(node.widgets.indexOf(combo), 0, btn);
        };
        moveBtnToFront();
        setTimeout(moveBtnToFront, 300);
    },
});
