# 🏆 Legend Ware v2

Legend Ware is a high-performance, GPU-accelerated Roblox triggerbot with a premium "FPS Cheat" style overlay. It uses ONNX for object detection and Windows API for 100% transparent visual interaction.

## 🚀 Quick Start (Bắt đầu nhanh)

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Bot
```bash
python bot_square.py
```

## 🎮 Game Settings (Cài đặt Game)
* **Mode**: You MUST use **Windowed** or **Borderless Window** in Roblox.
* **Overlay**: The bot will draw a Red Box (hitbox) and Green Crosshairs over your game.

## ⌨️ Keybinds (Phím tắt)
| Key | Action | Description |
| :--- | :--- | :--- |
| **`INSERT`** | Toggle Menu | Show or hide the configuration GUI. |
| **`C`** | Hold-to-Track | The bot only scans for targets while this key is HELD. |
| **`Q`** | Trigger Key | The bot automatically presses this when a target enters the zone. |

## ✨ Features (Tính năng)
* 🛠️ **Configurable UI**: Change hitbox size and keybinds on the fly.
* 🖥️ **True Transparency**: Uses Windows API (`pywin32`) for an invisible background.
* ⚡ **GPU Accelerated**: Powered by `onnxruntime-directml` for maximum FPS.
* 📈 **Prediction Line**: Shows estimated future position of the target.
* 🎯 **Square Hitbox**: Precise detection logic for specific zones.

## 🏮 Troubleshooting (Sửa lỗi)
* **Black Screen?**: This is fixed in v2. Ensure `pywin32` is installed.
* **Not Clicking?**: Make sure the game is focused and you are holding the correct key.
* **Low FPS?**: Ensure your GPU drivers are up to date.

---
*Created with ❤️ for Volleyball Legend Botting.*
