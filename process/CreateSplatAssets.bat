@echo off
set inputFile=%1
set folderName=%~n1_result

mkdir %folderName%

set ply=%folderName%\%~n1_output.ply
set geo=%folderName%\%~n1_geo.asset
set mesh=%folderName%\%~n1_mesh.asset
set order=%folderName%\%~n1_order.asset
set sh=%folderName%\%~n1_sh.asset


resplat2 %inputFile% %ply% %geo% %mesh% %order% %sh%