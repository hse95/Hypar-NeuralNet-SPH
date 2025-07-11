#!/bin/bash 

fail () { 
 echo Execution aborted. 
 read -n1 -r -p "Press any key to continue..." key 
 exit 1 
}

# "name" and "dirout" are named according to the testcase

export name=case
export dirout=${name}_out
export diroutdata=${dirout}/data

# "executables" are renamed and called from their directory

export dirbin=$HOME/DualSPHysics/bin/linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${dirbin}
export gencase="${dirbin}/GenCase_linux64"
export dualsphysicscpu="${dirbin}/DualSPHysics5.2CPU_linux64"
export dualsphysicsgpu="${dirbin}/DualSPHysics5.2_linux64"
export boundaryvtk="${dirbin}/BoundaryVTK_linux64"
export partvtk="${dirbin}/PartVTK_linux64"
export partvtkout="${dirbin}/PartVTKOut_linux64"
export measuretool="${dirbin}/MeasureTool_linux64"
export computeforces="${dirbin}/ComputeForces_linux64"
export isosurface="${dirbin}/IsoSurface_linux64"
export flowtool="${dirbin}/FlowTool_linux64"
export floatinginfo="${dirbin}/FloatingInfo_linux64"
export tracerparts="${dirbin}/TracerParts_linux64"

option=-1
 if [ -e $dirout ]; then
 while [ "$option" != 1 -a "$option" != 2 -a "$option" != 3 ] 
 do 

	echo -e "The folder "${dirout}" already exists. Choose an option.
  [1]- Delete it and continue.
  [2]- Execute post-processing.
  [3]- Abort and exit.
"
 read -n 1 option 
 done 
  else 
   option=1 
fi 

if [ $option -eq 1 ]; then
# "dirout" to store results is removed if it already exists
if [ -e ${dirout} ]; then rm -r ${dirout}; fi

# CODES are executed according the selected parameters of execution in this testcase

${gencase} ${name}_Def ${dirout}/${name} -save:all
if [ $? -ne 0 ] ; then fail; fi

${dualsphysicsgpu} -gpu -dbc ${dirout}/${name} ${dirout} -dirdataout data -svres
if [ $? -ne 0 ] ; then fail; fi

fi


# Post-Processing

# Export fluid particles
if [ $option -eq 2 -o $option -eq 1 ]; then
export dirout2=${dirout}/particles
${partvtk} -dirin ${diroutdata} -savevtk ${dirout2}/PartFluid -onlytype:-all,+fluid -onlypos:40:-2.5:-1:80:+2.5:20 -vars:-all,+vel,+press

# Export forces (mk + 9 + 1)
export dirout2=${dirout}/forces
${computeforces} -dirin ${diroutdata} -onlymk:15 -viscoauto -savecsv ${dirout2}/BW_Forces
if [ $? -ne 0 ] ; then fail; fi

# Post-Processing
export dirout2=${dirout}/pressure
${measuretool} -dirin ${diroutdata} -points /scratch/gpfs/hse/DualSPHysics/NNFSBW/probes/Rn05dr25.txt -onlytype:-all,+fluid -vars:-all,+press -savecsv ${dirout2}/PointsPressureOut
if [ $? -ne 0 ] ; then fail; fi

# Export Elevations
export dirout2=${dirout}/MeasureElevation
${measuretool} -dirin ${diroutdata} -filexml ${dirout}/case.xml -savecsv ${dirout2}/MeasureSWL -points /scratch/gpfs/hse/DualSPHysics/NNFSBW/probes/waterelevpoints.txt -vars:-all -height
if [ $? -ne 0 ] ; then fail; fi




# Export Piston VTK
# ${partvtk} -dirin ${diroutdata} -savevtk ${dirout2}/PartPiston -onlytype:-all,+moving
# if [ $? -ne 0 ] ; then fail; fi
# Resume Fluid
# ${partvtkout} -dirin ${diroutdata} -savevtk ${dirout2}/PartFluidOut -SaveResume ${dirout2}/_ResumeFluidOut
# if [ $? -ne 0 ] ; then fail; fi
# export dirout2=${dirout}/surface
# ${isosurface} -dirin ${diroutdata} -saveiso ${dirout2}/Surface 
# if [ $? -ne 0 ] ; then fail; fi

fi
if [ $option != 3 ];then
 echo All done
 else
 echo Execution aborted
fi

read -n1 -r -p "Press any key to continue..." key
