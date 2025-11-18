#!/bin/bash
# TensorBoard başlatma scripti

echo "🚀 Starting TensorBoard..."
echo "📊 Open browser: http://localhost:6006"
echo ""

tensorboard --logdir=runs --port=6006
