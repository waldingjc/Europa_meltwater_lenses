set myTime=%time::=.%
set myTime=%myTime:~0,-6%
set myTime=%myTime: =.% 

set myDate=%date%
set myDate=%myDate: =.%
set myDate=%myDate:/=.%

set myFolder=run_1_slab_LENS_r-triple_3030-revolve-2_%myDate%_time_%myTime%
echo %myFolder%

chdir runs
mkdir %myFolder%
chdir %myFolder%
mkdir data_dump
mkdir data_out

set run=Europa_Fractures_r
set domain=200kmx200kmx20km_LENS_REVOLVE_CHAMFER.3dm
set flaws=flaw_setup_lens_side-triple.txt
copy "C:\Coding-Walding\bin\%run%.exe"
copy "..\..\3dm2tin.exe" .
copy ..\..\run_1_slab_lens_single.bat .
copy ..\..\samg.in .
copy ..\..\3dms\%domain% .
copy ..\..\flaws\%flaws% .
copy ..\..\satstress_inputs.sat .

set fixed_flags=-M -load_rhino %domain% -allow_slaves_as_intersection_masters -bc Europa_BCT -solver_pardiso
set growth_flags=-dRTip -sif_DISC -quadratic -withqp -paris_critical -mixed_mode_failure_criterion -max_ext 1e2
set box_flags=-box_distance_multiplier 1 -box_distance_multiplierTOP 0 -box_distance_multiplierSIDES 2 -box_distance_multiplierBOTTOM 1
set misc_flags=-skip_zero_column -mesh_edge_curves -a 1

set material_flags=-ym 10e9 -pr 0.33 -uts 1.5e6 -kic 108e3 -beta_n 0 -beta_f 0
rem look into kic again --double check all physical parameters (tensile strength = 1/10 compressive)

set refine_flags=-refine 1 -fixed_volume_refine 5e4 -fixed_edge_refine 5e4
set mesh_flags=-meshing_flagsn "ic_geo_set_family_params INTLENS_G_S emax 50 internal_wall 1" -meshing_flagsn "ic_geo_set_family_params MATRIX_15 emax 5" -meshing_flagsn "ic_set_meshing_params gnat 0.1"
set frac_flags=-fracture_setup %flaws% 
rem -ffr 1e3

set data_flags=-vtk -dfnstats
set speed_flags=-no_vtk

set europa_flags=-tidal -time_code 4 64 1e7 1 -mesh_scale 1e-2 
set europa_layer_flags=-radial_layer 0 1500e3 1530e3 0.5 2 1500e3 1530e3 0.5 2 1420e3 1500e3 1e9 x 
set europa_region_flags=-regions 1530e3 1560e3 0.1 8 1e4 0 2 x 1530e3 1560e3 10 8 1e4 0 2 x y

set slab_flags=%europa_flags% %data_flags% -applied_load 1 -slabloc 30 30 -slabdim 200e3 200e3 20e3 -radgrav
rem %frac_flags% -overburden

set runtime_flags=-nr_of_tips 40 -min_ext 0 -max_it 4 -nostress -no_exit -no_mapping
set nuc_flags=-nucleation -nuc_flaw_size 1e3 -nucleation_max 3 -nuc_abs_spacing 10e3 -nuc_step_spacing_multiplier 0.95 -nucleation_max_multiplier 1 -step_nucleation 0
rem

cmd /k %run%.exe %fixed_flags% %growth_flags% %box_flags% %misc_flags% %material_flags% %refine_flags% %slab_flags% %runtime_flags% %nuc_flags% %mesh_flags% %data_flags%
rem %run%.exe %fixed_flags% %growth_flags% %box_flags% %misc_flags% %material_flags% %refine_flags% %slab_flags% %runtime_flags% %speed_flags% %frac_flags%


rem %europa_layer_flags% %europa_region_flags% -start_time 1e2 %data_flags% -nucleation_flaw_size_multiplier 0.95 -noprint_tip_info
rem -fixed-extension-w -skip_sifs_and_growth -bcp -1.e-5 -friction -uzawa_outer 2 -uzawa_inner 2 -rfs  -solver_max_cycles 1000

rem remove -no_mapping and add -ym_damage

we assume y is NorthSouth
gravity is 1.315ms-2
youngs mod is 16-17GPa (Neumeier 2018)
poissons ratio 0.35 (Ivins 2021)
we're not taking into account potential in situ stresses caused by the ice shelf itself
we need to take into account the size effect, and in general the effect of the size effect on the mechanical properties

radius of the inner surface is 1421km, radius of Europa is 1561km
therefore simulated crust thickness is 140km


original ==> material_flags=-ym 10e9 -pr 0.33 -ft 1e6 -beta_n 0 -beta_f 0 -kic 0.12e6
echo the time code syntax is number_of_europan_days || number_of_steps_for_that_amount || number_of_jovian_years || number_of_steps_for_that_amount

radial layer property codes::   0 -> youngs modulus
                                1 -> youngs modulus continuous
                                2 -> kic
                                3 -> kic continuous
                                when further necessary add to material.cpp