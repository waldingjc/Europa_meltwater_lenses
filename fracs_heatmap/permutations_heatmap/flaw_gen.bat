setlocal enabledelayedexpansion

for /L %%x in (0, 1, 6) do (
    for /L %%y in (0, 1, 6) do (
        for /L %%o in (0, 1, 2) do (
            if %%o==0 (set orientation_code=0 0 1)
            if %%o==0 (set orientation_name=hori)
            if %%o==1 (set orientation_code=1 0 0)
            if %%o==1 (set orientation_name=vert)
            if %%o==2 (set orientation_code=1 1 1)
            if %%o==2 (set orientation_name=diag)
            rem orientations; 0 - horizontal; 1 - vertical; 2 - diagonal

            rem set /a x_coord=%%x * 7500 - 45000
            rem set /a y_coord=%%y * 7500 - 45000
            rem 7.5 is 45 / 6, I would have rather written (z - 6) * (45 / 6)

            set /a x_coord=%%x * 15000 - 45000
            set /a y_coord=%%y * 15000 - 45000
            rem 15 is 45 / 3, I would have rather written (z - 3) * (45 / 3)

            (
                Echo f
                Echo -radius fixed 1000
                Echo -orientation fixed !orientation_code!
                Echo -location fixed !x_coord! !y_coord! 8500
            ) > "C:\Europa_Runs\flaws\permutations_fix\flaw_%%x-%%y_!orientation_name!.txt"
        )
    )
)