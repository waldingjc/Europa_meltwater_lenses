set myTime=%time::=.%
set myTime=%myTime:~0,-6%
set myTime=%myTime: =.% 

set myDate=%date%
set myDate=%myDate: =.%
set myDate=%myDate:/=.%

set myFolder=manynfrac_lens_run_very_tight_edge_hori_%myDate%_time_%myTime%
set run=Europa_Fractures_r

set domain=200kmx200kmx20km_LENS_REVOLVE_CHAMFER.3dm
set deg_north=30
set deg_east=100
set flaws=flaw_edge_vert_matrix_very_tight_hori

chdir runs
mkdir %myFolder%
chdir %myFolder%

copy ..\..\%run%.exe .
copy ..\..\3dm2tin.exe .
copy %0
copy ..\..\samg.in .
copy ..\..\3dms\%domain% .
copy ..\..\satstress_inputs.sat .
copy ..\..\flaws\%flaws%.txt

set fixed_flags=-M -load_rhino %domain% -allow_slaves_as_intersection_masters -bc Europa_BCT -solver_pardiso
set growth_flags=-dRTip -sif_DC -quadratic -withqp -paris_critical -mixed_mode_failure_criterion -max_ext 1e2
set box_flags=-box_distance_multiplier 1 -box_distance_multiplierTOP 0 -box_distance_multiplierSIDES 2 -box_distance_multiplierBOTTOM 1
set misc_flags=-skip_zero_column -mesh_edge_curves -a 1
set material_flags=-ym 10e9 -pr 0.33 -uts 1.5e6 -kic 108e3 -beta_n 0 -beta_f 0
set refine_flags=-refine 1 -fixed_volume_refine 5e4 -fixed_edge_refine 5e4
set mesh_flags=-meshing_flagsn "ic_geo_set_family_params INTLENS_G_S emax 50 internal_wall 1" -meshing_flagsn "ic_set_meshing_params gnat 0.1"
set runtime_flags=-nr_of_tips 10 -min_ext 0 -max_it 20 -nostress -no_exit -no_mapping -no_print_tip_info -no_vtk -dfnstats
set europa_flags=-tidal -time_code 1 128 1e7 1 -mesh_scale 1e-2
set slab_flags=-applied_load 1 -slabdim 200e3 200e3 20e3 -radgrav

set flags=%fixed_flags% %growth_flags% %box_flags% %misc_flags% %material_flags% %refine_flags% %mesh_flags% %runtime_flags% %europa_flags% %slab_flags%

set nuc_flags=-nucleation -nuc_flaw_size 1e3 -nucleation_max 40 -nuc_abs_spacing 10e3 -nuc_step_spacing_multiplier 0.95 -nucleation_max_multiplier 1.1 -step_nucleation 0

mkdir data_dump
mkdir data_out

cmd /k %run%.exe %flags% -slabloc %deg_north% %deg_east% -fracture_setup %flaws%.txt