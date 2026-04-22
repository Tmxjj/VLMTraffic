import re

with open('scripts/watermark_ui.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Clean presets and currentConfig
html = re.sub(r'const genBase[\s\S]*?};\n', 'let presets = {};\n', html)
html = html.replace('let currentConfig = JSON.parse(JSON.stringify(presets[currentScenario]));', 'let currentConfig = null;')

# 2. updateApproachSelector null check
html = html.replace('function updateApproachSelector() {', 'function updateApproachSelector() {\n            if (!currentConfig) return;')

# 3. Add autoSave and initConfig
init_str = """
        let autoSaveTimer = null;
        function scheduleAutoSave() {
            if (autoSaveTimer) clearTimeout(autoSaveTimer);
            autoSaveTimer = setTimeout(async () => {
                if (!currentConfig) return;
                try {
                    await fetch('/api/save_config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ config_key: currentScenario, config: currentConfig })
                    });
                    const btn = document.getElementById('saveBtn');
                    btn.textContent = "Saved";
                    setTimeout(() => btn.textContent = "Save to add_lane_watermarks.py (保存到脚本)", 1000);
                } catch(e) {}
            }, 300);
        }

        async function initConfig() {
            try {
                const res = await fetch('/api/get_config');
                presets = await res.json();
            } catch(e) {
                console.error(e);
            }
            if (presets[currentScenario]) {
                currentConfig = JSON.parse(JSON.stringify(presets[currentScenario]));
            } else {
                currentConfig = {};
            }
            updateApproachSelector();
            buildControls();
            draw();
        }
        
        initConfig(); // run on load
"""
# insert before updateApproachSelector
html = html.replace('function updateApproachSelector() {', init_str + '\n        function updateApproachSelector() {')

# 4. Modify scenarioSelect event listener
html = html.replace('currentConfig = JSON.parse(JSON.stringify(presets[currentScenario]));', 'if (presets[currentScenario]) currentConfig = JSON.parse(JSON.stringify(presets[currentScenario])); else currentConfig = {};')

# 5. Modify buildControls null check
html = html.replace('const cfg = currentConfig[currentApproach];', 'if (!currentConfig) return;\n            const cfg = currentConfig[currentApproach];')

# 6. Add scheduleAutoSave() to slider input
html = html.replace('draw();\n            };', 'draw();\n                scheduleAutoSave();\n            };')

# 7. Add scheduleAutoSave() to mouseup
html = html.replace('dragTarget = null);', 'dragTarget = null; scheduleAutoSave(); });')

with open('scripts/watermark_ui.html', 'w', encoding='utf-8') as f:
    f.write(html)
