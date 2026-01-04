# AI Compiler UI Architecture (v2)

This document outlines the "Avant-Garde Glass" design system and the architectural decisions behind the UI v2.

## 1. Design Philosophy
*   **Theme**: Dark Mode with Glassmorphism (`backdrop-filter: blur`).
*   **Font**: Inter (UI) + JetBrains Mono (Code/Terminal).
*   **Layout**: Strict 2-column grid (`2/3 Main`, `1/3 Sidebar`) for high readability.

## 2. The Grid System (`.cols-2-1`)
We use a global CSS utility class `.cols-2-1` to enforce a consistent layout across all steps.

```css
.cols-2-1 {
    display: grid;
    grid-template-columns: 2fr 1fr; /* 66% Content, 33% Sidebar */
    gap: 24px;
    align-items: start;
}
```

**Usage in Gradio:**
```python
with gr.Row(elem_classes=["cols-2-1"]):
    with gr.Column():
        # Main Content
    with gr.Column(elem_classes=["sticky-column"]):
        # Sidebar Settings
```

## 3. Z-Index Hierarchy (Critical)
To prevent overlapping issues between the Drawer, Dropdowns, and Sticky headers, we enforce a strict Z-index scale.

| Component | Z-Index | Note |
| :--- | :--- | :--- |
| **Drawer (`.drawer-content`)** | `2147483647` | Max 32-bit INT (Dominant). |
| **Dropdowns (`ul.options`)** | `5000` | Must float above cards. |
| **Sticky Sidebar** | `Default` | Standard document flow. |
| **Cards (`.premium-card`)** | `1` | Base layer. |

## 4. Theme Tokens & Classes
Use these utility classes in `engine/ui_v2/consts.py` to maintain consistency.

*   **.premium-card**: The base container with glass effect and border.
*   **.gr-input**: Standard styling for textboxes and selects.
*   **.primary-btn**: The gradient call-to-action button.
*   **.mono-text**: For logs, IDs, and code snippets.
*   **.badge-live** / **.badge-idle**: Status pillars.

## 5. Verification
Run these scripts to verify UI integrity after changes:
1.  **UI Classes Check**: `uv run python scripts/verify_ui_refactor.py`
2.  **Drawer Logic**: `uv run python scripts/verify_ui_drawer.py`
