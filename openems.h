/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPENEMS_H
#define OPENEMS_H

#include <sstream>
#if defined(_WIN32) && !defined(__GNUC__)
#include <Winsock2.h> // for struct timeval
#else
#include <sys/time.h>
#endif
#include <ctime>
#include <vector>

#include "openems_global.h"

#define OPENEMS_STAT_FILE "openEMS_stats.txt"
#define OPENEMS_RUN_STAT_FILE "openEMS_run_stats.txt"

//! Default number of timesteps between two evaluations of the end-criteria.
//! Measured cost, at one thread on three beds (a 41^3 and a 61^3 cell air box
//! and a shielded microstrip line): 0.7 to 1.3 timesteps per evaluation on this
//! build, so this default costs about 1% of the run time. An earlier
//! instrumented build that timed the parts separately put the same three beds
//! at 1.3 to 1.4, so treat 1 to 1.4% as the range. Most of that is not the
//! energy estimate itself (0.2 to 0.3 timesteps on that instrumented build)
//! but the extra IterateTS call that clamping the loop to the next evaluation
//! forces. The share is expected
//! to be worse where threads speed the update up, because the energy estimate
//! is a serial scalar pass over every cell while the update is SSE and
//! multi-threaded -- that is read off the code, not measured, since the machine
//! these numbers come from gave no thread speedup at all.
//! It has to be a number of timesteps and not a wall-clock period: sampling the
//! end-criteria on a wall-clock period makes the number of simulated timesteps
//! -- and with it the frequency resolution of every result -- depend on the
//! speed of the host.
#define OPENEMS_DEFAULT_ENDCRIT_CHECK_INTERVAL 100

class Operator;
class Engine;
class Engine_Interface_FDTD;
class ProcessingArray;
class TiXmlElement;
class TiXmlNode;
class ContinuousStructure;
class Engine_Interface_FDTD;
class Excitation;
class Engine_Ext_SteadyState;

double CalcDiffTime(timeval t1, timeval t2);
std::string FormatTime(int sec);

class OPENEMS_EXPORT openEMS
{
public:
	openEMS();
	virtual ~openEMS();

	virtual void showUsage();

	bool ParseFDTDSetup(std::string file);
	virtual bool Parse_XML_FDTDSetup(TiXmlElement* openEMSxml);
	virtual int SetupFDTD();
	virtual void RunFDTD();

	void Reset();

	void SetNumberOfTimeSteps(unsigned int val) {NrTS=val;}
	void SetEnableDumps(bool val) {Enable_Dumps=val;}
	//! Set the end-criteria: the energy decay at which the simulation stops (has to be <1)
	void SetEndCriteria(double val);
	//! Set the number of timesteps between two evaluations of the end-criteria (has to be >0)
	void SetEndCriteriaCheckInterval(unsigned int val);
	void SetOverSampling(int val) {m_OverSampling=val;}
	void SetCellConstantMaterial(bool val) {m_CellConstantMaterial=val;}

	void SetCylinderCoords(bool val) {CylinderCoords=val;}
	void SetupCylinderMultiGrid(std::vector<double> val) {m_CC_MultiGrid=val;}
	void SetupCylinderMultiGrid(std::string val);

	void SetTimeStepMethod(int val) {m_TS_method=val;}
	void SetTimeStep(double val) {m_TS=val;}
	void SetTimeStepFactor(double val) {m_TS_fac=val;}
	void SetMaxTime(double val) {m_maxTime=val;}
	//! Set the max. wall-clock run time of the FDTD loop in seconds (0 to disable)
	void SetMaxRunTime(double val) {m_maxRunTime=val;}

	// used by Python binding when running as a shared library
	void SetLibraryArguments(std::vector<std::string> allOptions);

	void SetNumberOfThreads(int val);

	void DebugMaterial() {DebugMat=true;}
	void DebugOperator() {DebugOp=true;}
	void DebugBox() {m_debugBox=true;}
	void DebugPEC() {m_debugPEC=true;}
	void DebugCSX() {m_debugCSX=true;}

	void Set_BC_Type(int idx, int type);
	int Get_BC_Type(int idx);
	void Set_BC_PML(int idx, unsigned int size);
	int Get_PML_Size(int idx);
	void Set_Mur_PhaseVel(int idx, double val);

	//! Get information about external libs used by openEMS
	static std::string GetExtLibsInfo(std::string prefix="\t");

	//! Get welcome screen for openEMS
	static void WelcomeScreen();

	//! Set this to about FDTD iteration process
	void SetAbort(bool val) {m_Abort=val;}
	//! Check for abort conditions
	bool CheckAbortCond();
	//! Check whether the max. wall-clock run time has been exceeded
	bool CheckRunTimeLimit(double t_run) const;

	//! Reason the FDTD iteration loop was left
	enum TerminationReason
	{
		Terminated_NotRun,       //!< RunFDTD has not been run (yet)
		Terminated_Converged,    //!< the end-criteria was reached
		Terminated_MaxTimesteps, //!< the max. number of timesteps was reached
		Terminated_MaxRunTime,   //!< the max. wall-clock run time was reached
		Terminated_Aborted       //!< aborted by SIGINT, an "ABORT" file or SetAbort()
	};

	//! Get the reason the last FDTD run was terminated
	TerminationReason GetTerminationReason() const {return m_TerminationReason;}
	//! Get the reason the last FDTD run was terminated as a short, stable keyword
	std::string GetTerminationReasonString() const;

	void SetGaussExcite(double f0, double fc);
	void SetSinusExcite(double f0);
	void SetDiracExcite(double f_max);
	void SetStepExcite(double f_max);
	void SetCustomExcite(std::string str, double f0, double fmax);

	Excitation* InitExcitation();

	//! Set the geometry, taking ownership: \a csx is destroyed by this class,
	//! as is any structure set before. \sa GetCSX
	void SetCSX(ContinuousStructure* csx);
	//! Get the geometry, which stays owned by this class. \sa SetCSX
	ContinuousStructure* GetCSX() const;

	Engine_Interface_FDTD* NewEngineInterface(int multigridlevel = 0);

	void SetVerboseLevel(int level);

	bool Write2XML(TiXmlNode* rootNode);
	bool Write2XML(std::string file);
	bool ReadFromXML(std::string file);

protected:
	void collectCommandLineArguments();

	bool CylinderCoords;
	std::vector<double> m_CC_MultiGrid;

	ContinuousStructure* m_CSX;

	//! Number of Timesteps
	unsigned int NrTS;
	int m_TS_method;
	double m_TS;
	double m_TS_fac;
	double m_maxTime;
	double m_maxRunTime;

	// some command line flags
	bool Enable_Dumps;
	bool DebugMat;
	bool DebugOp;
	bool m_debugCSX;
	bool m_DumpStats;
	bool m_debugBox, m_debugPEC, m_no_simulation;

	double endCrit;
	unsigned int m_endCritCheckInterval;
	int m_OverSampling;
	bool m_CellConstantMaterial;
	Operator* FDTD_Op;
	Engine* FDTD_Eng;
	Engine_Ext_SteadyState* Eng_Ext_SSD; //!< non-owning observer; owned/deleted by the engine (m_Eng_exts)
	ProcessingArray* PA;

	Excitation* m_Exc;

	bool m_Abort;
	TerminationReason m_TerminationReason;

#ifdef MPI_SUPPORT
	enum EngineType {EngineType_Basic, EngineType_SSE, EngineType_SSE_Compressed, EngineType_Multithreaded, EngineType_MPI};
#else
	enum EngineType {EngineType_Basic, EngineType_SSE, EngineType_SSE_Compressed, EngineType_Multithreaded};
#endif
	EngineType m_engine;
	unsigned int m_engine_numThreads;

	//! Setup an operator matching the requested engine
	virtual bool SetupOperator();

	//! Read boundary conditions from xml element and apply to FDTD operator
	bool SetupBoundaryConditions();
	int m_BC_type[6];
	unsigned int m_PML_size[6];
	double m_Mur_v_ph[6];

	//! Setup local absorbing boundary conditions
	void SetupAbsorbingSheets();

	//! Check whether or not the FDTD-Operator has to store material data.
	bool SetupMaterialStorages();

	//! Setup all processings.
	virtual bool SetupProcessing();

	//! Dump statistics to file
	virtual bool DumpStatistics(const std::string& filename, double time);

	//! Dump run statistivs to file
	virtual bool InitRunStatistics(const std::string& filename);
	virtual bool DumpRunStatistics(const std::string& filename, double time, unsigned int ts, double speed, double energy);
};

#endif // OPENEMS_H
