setlocal enabledelayedexpansion

set myTime=%time::=.%
set myTime=%myTime:~0,-6%
set myTime=%myTime: =.% 

set myDate=%date%
set myDate=%myDate: =.%
set myDate=%myDate:/=.%

set myFolder=quant_fix_nolens-longi_%myDate%_time_%myTime%
set run=Europa_Fractures_r
set domain=200kmx200kmx20km.3dm
set deg_north=30

chdir runs
mkdir %myFolder%
chdir %myFolder%

copy ..\..\%run%.exe .
copy ..\..\3dm2tin.exe .
copy ..\..\run_Quant_lens.bat .
copy ..\..\samg.in .
copy ..\..\3dms\%domain% .
copy ..\..\satstress_inputs.sat .

set fixed_flags=-M -load_rhino %domain% -allow_slaves_as_intersection_masters -bc Europa_BCT -solver_pardiso
set growth_flags=-dRTip -sif_DC -quadratic -withqp -paris_critical -mixed_mode_failure_criterion -max_ext 1e2
set box_flags=-box_distance_multiplier 1 -box_distance_multiplierTOP 0 -box_distance_multiplierSIDES 2 -box_distance_multiplierBOTTOM 1
set misc_flags=-skip_zero_column -mesh_edge_curves -a 1
set material_flags=-ym 10e9 -pr 0.33 -uts 1.5e6 -kic 108e3 -beta_n 0 -beta_f 0
set refine_flags=-refine 1 -fixed_volume_refine 5e4 -fixed_edge_refine 5e4
set mesh_flags=-meshing_flagsn "ic_geo_set_family_params INTLENS_G_S emax 50 internal_wall 1" -meshing_flagsn "ic_set_meshing_params gnat 0.1"
set runtime_flags=-nr_of_tips 20 -min_ext 0 -max_it 2 -nostress -no_exit -no_mapping -print_tip_info -vtk -dfnstats
set europa_flags=-tidal -time_code 1 1 1e7 1 -mesh_scale 1e-2
set slab_flags=-applied_load 1 -slabdim 200e3 200e3 20e3 -radgrav

set flags=%fixed_flags% %growth_flags% %box_flags% %misc_flags% %material_flags% %refine_flags% %mesh_flags% %runtime_flags% %europa_flags% %slab_flags%

for /L %%e in (0, 5, 180) do (
    for /L %%x in (0, 1, 0) do (
        for /L %%y in (0, 1, 0) do (
            for /L %%o in (0, 1, 2) do (
                if %%o==0 (set orientation_name=hori)
                if %%o==1 (set orientation_name=vert)
                if %%o==2 (set orientation_name=diag)

                set flaw_code=%%x-%%y_!orientation_name!
                set flaws=flaw_!flaw_code!.txt

                set run_folder=run_%%e_!flaw_code!
                mkdir !run_folder!
                chdir !run_folder!
                mkdir data_dump
                mkdir data_out

                copy ..\%run%.exe .
                copy ..\%domain% .
                copy ..\3dm2tin.exe .
                copy ..\samg.in .
                copy ..\satstress_inputs.sat .
                copy ..\run_Quant_lens.bat .
                copy ..\..\..\flaws\permutations_1_nolens\!flaws! .

                %run%.exe %flags% -slabloc %deg_north% %%e -fracture_setup !flaws!
                rem echo %flags% -slabloc %deg_north% %%e -fracture_setup !flaws! >> run_flags.txt
                
                del %run%.exe
                del %domain%
                del 3dm2tin.exe
                del samg.in
                del satstress_inputs.sat
                del run_Quant_lens.bat
                del !flaws!

                chdir ..
            )
        )
    )
)
