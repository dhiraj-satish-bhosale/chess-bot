$logFile = "c:\Users\niraj\OneDrive\Desktop\Chess Bot\chess_bot_updated\chess_bot\lichess-bot\lichess_bot_auto_logs\lichess-bot.log"
$outLog = "c:\Users\niraj\OneDrive\Desktop\Chess Bot\chess_bot_updated\chess_bot\training_output.log"

Write-Output "$(Get-Date): Monitoring lichess-bot.log for 35/35 games..." | Out-File -FilePath $outLog

while ($true) {
    # Check if we hit 35/35 in the log file
    $found = Select-String -Path $logFile -Pattern "Total games played today: 35/35" -Quiet
    if ($found) {
        Write-Output "$(Get-Date): Hit 35/35 games! Killing lichess-bot and starting training..." | Out-File -FilePath $outLog -Append
        
        # Kill the bot process
        $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object CommandLine -match "lichess-bot.py"
        foreach ($p in $processes) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }

        # Run training
        cd "c:\Users\niraj\OneDrive\Desktop\Chess Bot\chess_bot_updated\chess_bot"
        python scripts/train_joint.py --resume models/checkpoints/alphazero_distilled.pt --data data/train_elite_games.npz data/train_puzzles_large.npz --epochs 50 2>&1 | Out-File -FilePath $outLog -Append
        
        Write-Output "$(Get-Date): Training complete." | Out-File -FilePath $outLog -Append
        break
    }
    
    # Check if process died for another reason
    $process = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object CommandLine -match "lichess-bot.py"
    if (-not $process) {
        Write-Output "$(Get-Date): lichess-bot.py has stopped unexpectedly. Starting training anyway..." | Out-File -FilePath $outLog -Append
        cd "c:\Users\niraj\OneDrive\Desktop\Chess Bot\chess_bot_updated\chess_bot"
        python scripts/train_joint.py --resume models/checkpoints/alphazero_distilled.pt --data data/train_elite_games.npz data/train_puzzles_large.npz --epochs 50 2>&1 | Out-File -FilePath $outLog -Append
        Write-Output "$(Get-Date): Training complete." | Out-File -FilePath $outLog -Append
        break
    }
    
    Start-Sleep -Seconds 15
}
