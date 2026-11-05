# Railroads

A mechanics-first railroad tycoon game.

Run the game:

```bash
uv run railroads
```

Run the tests and linter:

```bash
uv run pytest
uv run ruff check
```

`src/railroads/sim` is the game logic and must never import the UI (ruff enforces this).
`src/railroads/ui` draws the game with Dear ImGui (imgui-bundle).
