/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "PreProcessor.h"
#include "Reconstructors.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Ply.h"
#include "VertexFactory.h"
#include "Image.h"
#include "RegularGrid.h"
#include "DataStream.imp.h"

#define DEFAULT_DIMENSION 3

cmdLineParameter< char* >
	In( "in" ) ,
	Out( "out" ) ,
	TempDir( "tempDir" ) ,
	Grid( "grid" ) ,
	Tree( "tree" ) ,
	Envelope( "envelope" ) ,
	Transform( "xForm" );

cmdLineReadable
	Performance( "performance" ) ,
	ShowResidual( "showResidual" ) ,
	PolygonMesh( "polygonMesh" ) ,
	NonManifold( "nonManifold" ) ,
	ASCII( "ascii" ) ,
	Density( "density" ) ,
	LinearFit( "linearFit" ) ,
	PrimalGrid( "primalGrid" ) ,
	ExactInterpolation( "exact" ) ,
	Colors( "colors" ) ,
	InCore( "inCore" ) ,
	NoDirichletErode( "noErode" ) ,
	Gradients( "gradients" ) ,
	Verbose( "verbose" );

cmdLineParameter< int >
#ifndef FAST_COMPILE
	Degree( "degree" , Reconstructor::Poisson::DefaultFEMDegree ) ,
#endif // !FAST_COMPILE
	Depth( "depth" , 8 ) ,
	KernelDepth( "kernelDepth" ) ,
	SolveDepth( "solveDepth" ) ,
	EnvelopeDepth( "envelopeDepth" ) ,
	Iters( "iters" , 8 ) ,
	FullDepth( "fullDepth" , 5 ) ,
	BaseDepth( "baseDepth" ) ,
	BaseVCycles( "baseVCycles" , 1 ) ,
#ifndef FAST_COMPILE
	BType( "bType" , Reconstructor::Poisson::DefaultFEMBoundary+1 ) ,
#endif // !FAST_COMPILE
	MaxMemoryGB( "maxMemory" , 0 ) ,
#ifdef _OPENMP
	ParallelType( "parallel" , (int)ThreadPool::OPEN_MP ) ,
#else // !_OPENMP
	ParallelType( "parallel" , (int)ThreadPool::THREAD_POOL ) ,
#endif // _OPENMP
	ScheduleType( "schedule" , (int)ThreadPool::DefaultSchedule ) ,
	ThreadChunkSize( "chunkSize" , (int)ThreadPool::DefaultChunkSize ) ,
	Threads( "threads" , (int)std::thread::hardware_concurrency() );

cmdLineParameter< float >
	DataX( "data" , 32.f ) ,
	SamplesPerNode( "samplesPerNode" , 1.5f ) ,
	Scale( "scale" , 1.1f ) ,
	Width( "width" , 0.f ) ,
	Confidence( "confidence" , 0.f ) ,
	ConfidenceBias( "confidenceBias" , 0.f ) ,
	CGSolverAccuracy( "cgAccuracy" , 1e-3f ) ,
	LowDepthCutOff( "lowDepthCutOff" , 0.f ) ,
	PointWeight( "pointWeight" );

cmdLineReadable* params[] =
{
#ifndef FAST_COMPILE
	&Degree , &BType ,
#endif // !FAST_COMPILE
	&In , &Depth , &Out , &Transform ,
	&SolveDepth ,
	&Envelope ,
	&Width ,
	&Scale , &Verbose , &CGSolverAccuracy ,
	&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
	&EnvelopeDepth ,
	&NoDirichletErode ,
	&ConfidenceBias ,
	&BaseDepth , &BaseVCycles ,
	&PointWeight ,
	&Grid , &Threads ,
	&Tree ,
	&Density ,
	&FullDepth ,
	&Iters ,
	&DataX ,
	&Colors ,
	&Gradients ,
	&LinearFit ,
	&PrimalGrid ,
	&TempDir ,
	&ExactInterpolation ,
	&Performance ,
	&MaxMemoryGB ,
	&InCore ,
	&ParallelType ,
	&ScheduleType ,
	&ThreadChunkSize ,
	&LowDepthCutOff ,
	NULL
};

void ShowUsage(char* ex)
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input points>\n" , In.name );
	printf( "\t[--%s <input envelope>\n" , Envelope.name );
	printf( "\t[--%s <ouput triangle mesh>]\n" , Out.name );
	printf( "\t[--%s <ouput grid>]\n" , Grid.name );
	printf( "\t[--%s <ouput fem tree>]\n" , Tree.name );
#ifndef FAST_COMPILE
	printf( "\t[--%s <b-spline degree>=%d]\n" , Degree.name , Degree.value );
	printf( "\t[--%s <boundary type>=%d]\n" , BType.name , BType.value );
	for( int i=0 ; i<BOUNDARY_COUNT ; i++ ) printf( "\t\t%d] %s\n" , i+1 , BoundaryNames[i] );
#endif // !FAST_COMPILE
	printf( "\t[--%s <maximum reconstruction depth>=%d]\n" , Depth.name , Depth.value );
	printf( "\t[--%s <maximum solution depth>=%d]\n" , SolveDepth.name , SolveDepth.value );
	printf( "\t[--%s <grid width>]\n" , Width.name );
	printf( "\t[--%s <full depth>=%d]\n" , FullDepth.name , FullDepth.value );
	printf( "\t[--%s <envelope depth>=%d]\n" , EnvelopeDepth.name , EnvelopeDepth.value );
	printf( "\t[--%s <coarse MG solver depth>]\n" , BaseDepth.name );
	printf( "\t[--%s <coarse MG solver v-cycles>=%d]\n" , BaseVCycles.name , BaseVCycles.value );
	printf( "\t[--%s <scale factor>=%f]\n" , Scale.name , Scale.value );
	printf( "\t[--%s <minimum number of samples per node>=%f]\n" , SamplesPerNode.name, SamplesPerNode.value );
	printf( "\t[--%s <interpolation weight>=%.3e * <b-spline degree>]\n" , PointWeight.name , Reconstructor::Poisson::WeightMultiplier * Reconstructor::Poisson::DefaultFEMDegree );
	printf( "\t[--%s <iterations>=%d]\n" , Iters.name , Iters.value );
	printf( "\t[--%s]\n" , ExactInterpolation.name );
	printf( "\t[--%s <pull factor>=%f]\n" , DataX.name , DataX.value );
	printf( "\t[--%s]\n" , Colors.name );
	printf( "\t[--%s]\n" , Gradients.name );
	printf( "\t[--%s <num threads>=%d]\n" , Threads.name , Threads.value );
	printf( "\t[--%s <parallel type>=%d]\n" , ParallelType.name , ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , ScheduleType.name , ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , ThreadChunkSize.name , ThreadChunkSize.value );
	printf( "\t[--%s <low depth cut-off>=%f]\n" , LowDepthCutOff.name , LowDepthCutOff.value );
	printf( "\t[--%s <normal confidence exponent>=%f]\n" , Confidence.name , Confidence.value );
	printf( "\t[--%s <normal confidence bias exponent>=%f]\n" , ConfidenceBias.name , ConfidenceBias.value );
	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s <cg solver accuracy>=%g]\n" , CGSolverAccuracy.name , CGSolverAccuracy.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , MaxMemoryGB.name , MaxMemoryGB.value );
	printf( "\t[--%s]\n" , NoDirichletErode.name );
	printf( "\t[--%s]\n" , Performance.name );
	printf( "\t[--%s]\n" , Density.name );
	printf( "\t[--%s]\n" , LinearFit.name );
	printf( "\t[--%s]\n" , PrimalGrid.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , TempDir.name );
	printf( "\t[--%s]\n" , InCore.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool HasGradients , bool HasDensity >
void WriteMesh
(
	bool inCore ,
	Reconstructor::Implicit< Real , Dim , FEMSig > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii
)
{
	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , HasGradients , HasDensity >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory();

	// A backing stream for the vertices
	Reconstructor::OutputInputFactoryTypeStream< Factory > vertexStream( factory , inCore , false , std::string( "v_" ) );
	Reconstructor::OutputInputFaceStream< Dim-1 > faceStream( inCore , true , std::string( "f_" ) );

	{
		// The wrapper converting native to output types
		typename VInfo::StreamWrapper _vertexStream( vertexStream , factory() );

		// Extract the level set
		implicit.extractLevelSet( _vertexStream , faceStream , meParams );
	}

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	PLY::Write< Factory , node_index_type , Real , Dim >( fileName , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ascii ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , typename AuxDataFactory , bool HasGradients , bool HasDensity >
void WriteMeshWithData
(
	const AuxDataFactory &auxDataFactory ,
	bool inCore ,
	Reconstructor::Implicit< Real , Dim , FEMSig , typename AuxDataFactory::VertexType > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii
)
{
	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexWithDataInfo< Real , Dim , AuxDataFactory , HasGradients , HasDensity >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory( auxDataFactory );

	// A backing stream for the vertices
	Reconstructor::OutputInputFactoryTypeStream< Factory > vertexStream( factory , inCore , false , std::string( "v_" ) );
	Reconstructor::OutputInputFaceStream< Dim-1 > faceStream( inCore , true , std::string( "f_" ) );

	{
		// The wrapper converting native to output types
		typename VInfo::StreamWrapper _vertexStream( vertexStream , factory() );

		// Extract the level set
		implicit.extractLevelSet( _vertexStream , faceStream , meParams );
	}

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	PLY::Write< Factory , node_index_type , Real , Dim >( fileName , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ascii ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
}

template< class Real , unsigned int Dim , unsigned int FEMSig , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	static const bool HasAuxData = !std::is_same< AuxDataFactory , VertexFactory::EmptyFactory< Real > >::value;

	///////////////
	// Types --> //
	typedef IsotropicUIntPack< Dim , FEMSig > Sigs;
	using namespace VertexFactory;

	// The factory for constructing an input sample's data
	typedef typename std::conditional< HasAuxData , Factory< Real , NormalFactory< Real , Dim > , AuxDataFactory > , NormalFactory< Real , Dim > >::type InputSampleDataFactory;

	// The factory for constructing an input sample
	typedef Factory< Real , PositionFactory< Real , Dim > , InputSampleDataFactory >  InputSampleFactory;

	typedef InputDataStream< typename InputSampleFactory::VertexType > InputPointStream;

	// The type storing the reconstruction solution (depending on whether auxiliary data is provided or not)
	using Implicit = typename std::conditional< HasAuxData , Reconstructor::Poisson::Implicit< Real , Dim , FEMSig , typename AuxDataFactory::VertexType > , Reconstructor::Poisson::Implicit< Real , Dim , FEMSig > >::type;
	// <-- Types //
	///////////////

	if( Verbose.set )
	{
		std::cout << "*************************************************************" << std::endl;
		std::cout << "*************************************************************" << std::endl;
		std::cout << "** Running Screened Poisson Reconstruction (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "*************************************************************" << std::endl;
		std::cout << "*************************************************************" << std::endl;
		if( !Threads.set ) std::cout << "Running with " << Threads.value << " threads" << std::endl;

		char str[1024];
		for( int i=0 ; params[i] ; i++ ) if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( strlen( str ) ) std::cout << "\t--" << params[i]->name << " " << str << std::endl;
			else                std::cout << "\t--" << params[i]->name << std::endl;
		}
	}

	Profiler profiler(20);
	Implicit *implicit = NULL;
	typename Reconstructor::Poisson::SolutionParameters< Real > sParams;
	Reconstructor::LevelSetExtractionParameters meParams;

	sParams.verbose = Verbose.set;
	sParams.dirichletErode = !NoDirichletErode.set;
	sParams.outputDensity = Density.set;
	sParams.exactInterpolation = ExactInterpolation.set;
	sParams.showResidual = ShowResidual.set;
	sParams.scale = (Real)Scale.value;
	sParams.confidence = (Real)Confidence.value;
	sParams.confidenceBias = (Real)ConfidenceBias.value;
	sParams.lowDepthCutOff = (Real)LowDepthCutOff.value;
	sParams.width = (Real)Width.value;
	sParams.pointWeight = (Real)PointWeight.value;
	sParams.samplesPerNode = (Real)SamplesPerNode.value;
	sParams.cgSolverAccuracy = (Real)CGSolverAccuracy.value;
	sParams.depth = (unsigned int)Depth.value;
	sParams.baseDepth = (unsigned int)BaseDepth.value;
	sParams.solveDepth = (unsigned int)SolveDepth.value;
	sParams.fullDepth = (unsigned int)FullDepth.value;
	sParams.kernelDepth = (unsigned int)KernelDepth.value;
	sParams.envelopeDepth = (unsigned int)EnvelopeDepth.value;
	sParams.baseVCycles = (unsigned int)BaseVCycles.value;
	sParams.iters = (unsigned int)Iters.value;

	meParams.linearFit = LinearFit.set;
	meParams.outputGradients = Gradients.set;
	meParams.forceManifold = !NonManifold.set;
	meParams.polygonMesh = PolygonMesh.set;
	meParams.verbose = Verbose.set;

	double startTime = Time();

	InputSampleFactory *_inputSampleFactory;
	if constexpr( HasAuxData ) _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , InputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory ) );
	else _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , VertexFactory::NormalFactory< Real , Dim >() );
	InputSampleFactory &inputSampleFactory = *_inputSampleFactory;
	XForm< Real , Dim+1 > toModel = XForm< Real , Dim+1 >::Identity();

	// Read in the transform, if we want to apply one to the points before processing
	if( Transform.set )
	{
		FILE* fp = fopen( Transform.value , "r" );
		if( !fp ) WARN( "Could not read x-form from: " , Transform.value );
		else
		{
			for( int i=0 ; i<Dim+1 ; i++ ) for( int j=0 ; j<Dim+1 ; j++ )
			{
				float f;
				if( fscanf( fp , " %f " , &f )!=1 ) ERROR_OUT( "Failed to read xform" );
				toModel(i,j) = (Real)f;
			}
			fclose( fp );
		}
	}
	std::vector< typename InputSampleFactory::VertexType > inCorePoints;
	InputPointStream *pointStream;

	// Get the point stream
	{
		profiler.reset();
		char *ext = GetFileExtension( In.value );

		if( InCore.set )
		{
			InputPointStream *_pointStream;
			if     ( !strcasecmp( ext , "bnpts" ) ) _pointStream = new BinaryInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) _pointStream = new    PLYInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else                                    _pointStream = new  ASCIIInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			typename InputSampleFactory::VertexType p = inputSampleFactory();
			while( _pointStream->read( p ) ) inCorePoints.push_back( p );
			delete _pointStream;

			pointStream = new VectorBackedInputDataStream< typename InputSampleFactory::VertexType >( inCorePoints );
		}
		else
		{
			if     ( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) pointStream = new    PLYInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else                                    pointStream = new  ASCIIInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
		}
		delete[] ext;
	}

	typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim > *envelopeMesh = NULL;
	if( Envelope.set )
	{
		envelopeMesh = new typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim >();
		{
			std::vector< std::vector< int > > polygons;
			std::vector< std::string > comments;
			int file_type;
			PLY::ReadPolygons( Envelope.value , PositionFactory< Real , Dim >() , envelopeMesh->vertices , polygons , file_type , comments );
			envelopeMesh->simplices.resize( polygons.size() );
			for( int i=0 ; i<polygons.size() ; i++ )
				if( polygons[i].size()!=Dim ) ERROR_OUT( "Not a simplex" );
				else for( int j=0 ; j<Dim ; j++ ) envelopeMesh->simplices[i][j] = polygons[i][j];
		}
	}

	// A wrapper class to realize InputDataStream< SampleType > as an InputSampleStream
	struct _InputSampleStream : public Reconstructor::InputSampleStream< Real , Dim >
	{
		typedef Reconstructor::Normal< Real , Dim > DataType;
		typedef VectorTypeUnion< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;
		_InputSampleStream( _InputPointStream &pointStream ) : pointStream( pointStream )
		{
			scratch = SampleType( Reconstructor::Position< Real , Dim >() , Reconstructor::Normal< Real , Dim >() );
		}
		void reset( void ){ pointStream.reset(); }
		bool base_read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n ) 
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>();
			return ret;
		}
	};

	// A wrapper class to realize InputDataStream< SampleType > as an InputSampleWithDataStream
	struct _InputSampleWithDataStream : public Reconstructor::InputSampleWithDataStream< Real , Dim , typename AuxDataFactory::VertexType >
	{
		typedef VectorTypeUnion< Real , Reconstructor::Normal< Real , Dim > , typename AuxDataFactory::VertexType > DataType;
		typedef VectorTypeUnion< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;
		_InputSampleWithDataStream( _InputPointStream &pointStream , typename AuxDataFactory::VertexType zero ) : Reconstructor::InputSampleWithDataStream< Real , Dim , typename AuxDataFactory::VertexType >( zero ) , pointStream( pointStream )
		{
			scratch = SampleType( Reconstructor::Position< Real , Dim >() , DataType( Reconstructor::Normal< Real , Dim >() , zero ) );
		}
		void reset( void ){ pointStream.reset(); }
		bool base_read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n , typename AuxDataFactory::VertexType &d ) 
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>().template get<0>() , d = scratch.template get<1>().template get<1>();
			return ret;
		}
	};

	if( Transform.set and envelopeMesh ) for( unsigned int i=0 ; i<envelopeMesh->vertices.size() ; i++ ) envelopeMesh->vertices[i] = toModel * envelopeMesh->vertices[i];
	if constexpr( HasAuxData )
	{
		_InputSampleWithDataStream sampleStream( *pointStream , auxDataFactory() );

		if( Transform.set )
		{
			Reconstructor::TransformedInputSampleWithDataStream< Real , Dim , typename AuxDataFactory::VertexType > _sampleStream( toModel , sampleStream );
			implicit = new typename Reconstructor::Poisson::Implicit< Real , Dim , FEMSig , typename AuxDataFactory::VertexType >( _sampleStream , sParams , envelopeMesh );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit = new typename Reconstructor::Poisson::Implicit< Real , Dim , FEMSig , typename AuxDataFactory::VertexType >( sampleStream , sParams , envelopeMesh );
	}
	else
	{
		_InputSampleStream sampleStream( *pointStream );

		if( Transform.set )
		{
			Reconstructor::TransformedInputSampleStream< Real , Dim > _sampleStream( toModel , sampleStream );
			implicit = new typename Reconstructor::Poisson::Implicit< Real , Dim , FEMSig >( _sampleStream , sParams , envelopeMesh );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit = new typename Reconstructor::Poisson::Implicit< Real , Dim , FEMSig >( sampleStream , sParams , envelopeMesh );
	}

	delete pointStream;
	delete _inputSampleFactory;
	delete envelopeMesh;

	if constexpr( HasAuxData ) if( implicit->auxData ) implicit->weightAuxDataByDepth( (Real)DataX.value );

	if( Tree.set )
	{
		FILE* fp = fopen( Tree.value , "wb" );
		if( !fp ) ERROR_OUT( "Failed to open file for writing: " , Tree.value );
		FileStream fs(fp);
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		implicit->tree.write( fs , implicit->unitCubeToModel.inverse() , false );
		implicit->solution.write( fs );
		if constexpr( HasAuxData ) if( implicit->auxData ) implicit->auxData->write( fs );
		if( implicit->density ) implicit->density->write( fs );
		fclose( fp );
	}

	if( Grid.set )
	{
		int res = 0;
		profiler.reset();
		Pointer( Real ) values = implicit->tree.template regularGridEvaluate< true >( implicit->solution , res , -1 , PrimalGrid.set );
		if( Verbose.set ) std::cout << "Got grid: " << profiler << std::endl;
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );

		unsigned int _res[Dim];
		for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
		RegularGrid< Real , Dim >::Write( Grid.value , _res , values , implicit->unitCubeToModel * voxelToUnitCube );

		DeletePointer( values );
	}


	if( Out.set )
		if( Dim==2 || Dim==3 )
		{
			// Create the output mesh
			char tempHeader[2048];
			{
				char tempPath[1024];
				tempPath[0] = 0;
				if( TempDir.set ) strcpy( tempPath , TempDir.value );
				else SetTempDirectory( tempPath , sizeof(tempPath) );
				if( strlen(tempPath)==0 ) sprintf( tempPath , ".%c" , FileSeparator );
				if( tempPath[ strlen( tempPath )-1 ]==FileSeparator ) sprintf( tempHeader , "%sPR_" , tempPath );
				else                                                  sprintf( tempHeader , "%s%cPR_" , tempPath , FileSeparator );
			}

			XForm< Real , Dim+1 > pXForm = implicit->unitCubeToModel;
			XForm< Real , Dim > nXForm = XForm< Real , Dim >( pXForm ).inverse().transpose();

			if( Gradients.set )
			{
				if( Density.set )
				{
					if constexpr( HasAuxData ) WriteMeshWithData< Real , Dim , FEMSig , AuxDataFactory , true , true >( auxDataFactory , InCore.set , *implicit , meParams , Out.value , ASCII.set );
					else                       WriteMesh        < Real , Dim , FEMSig ,                  true , true >(                  InCore.set , *implicit , meParams , Out.value , ASCII.set );
				}
				else
				{
					if constexpr( HasAuxData ) WriteMeshWithData< Real , Dim , FEMSig , AuxDataFactory , true , false >( auxDataFactory , InCore.set , *implicit , meParams , Out.value , ASCII.set );
					else                       WriteMesh        < Real , Dim , FEMSig ,                  true , false >(                  InCore.set , *implicit , meParams , Out.value , ASCII.set );
				}
			}
			else
			{
				if( Density.set )
				{
					if constexpr( HasAuxData ) WriteMeshWithData< Real , Dim , FEMSig , AuxDataFactory , false , true >( auxDataFactory , InCore.set , *implicit , meParams , Out.value , ASCII.set );
					else                       WriteMesh        < Real , Dim , FEMSig ,                  false , true >(                  InCore.set , *implicit , meParams , Out.value , ASCII.set );
				}
				else
				{
					if constexpr( HasAuxData ) WriteMeshWithData< Real , Dim , FEMSig , AuxDataFactory , false , false >( auxDataFactory , InCore.set , *implicit , meParams , Out.value , ASCII.set );
					else                       WriteMesh        < Real , Dim , FEMSig ,                  false , false >(                  InCore.set , *implicit , meParams , Out.value , ASCII.set );
				}
			}
		}
		else WARN( "Mesh extraction is only supported in dimensions 2 and 3" );

	if( Verbose.set ) std::cout << "#          Total Solve: " << Time()-startTime << " (s), " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
	delete implicit;
}

#ifndef FAST_COMPILE
template< unsigned int Dim , class Real , BoundaryType BType , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	switch( Degree.value )
	{
		case 1: return Execute< Real , Dim , FEMDegreeAndBType< 1 , BType >::Signature >( auxDataFactory );
		case 2: return Execute< Real , Dim , FEMDegreeAndBType< 2 , BType >::Signature >( auxDataFactory );
//		case 3: return Execute< Real , Dim , FEMDegreeAndBType< 3 , BType >::Signature >( auxDataFactory );
//		case 4: return Execute< Real , Dim , FEMDegreeAndBType< 4 , BType >::Signature >( auxDataFactory );
		default: ERROR_OUT( "Only B-Splines of degree 1 - 2 are supported" );
	}
}

template< unsigned int Dim , class Real , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	switch( BType.value )
	{
		case BOUNDARY_FREE+1:      return Execute< Dim , Real , BOUNDARY_FREE      >( auxDataFactory );
		case BOUNDARY_NEUMANN+1:   return Execute< Dim , Real , BOUNDARY_NEUMANN   >( auxDataFactory );
		case BOUNDARY_DIRICHLET+1: return Execute< Dim , Real , BOUNDARY_DIRICHLET >( auxDataFactory );
		default: ERROR_OUT( "Not a valid boundary type: " , BType.value );
	}
}
#endif // !FAST_COMPILE

int main( int argc , char* argv[] )
{
	Timer timer;
#ifdef ARRAY_DEBUG
	WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG
	cmdLineParse( argc-1 , &argv[1] , params );
	if( MaxMemoryGB.value>0 ) SetPeakMemoryMB( MaxMemoryGB.value<<10 );
	ThreadPool::DefaultChunkSize = ThreadChunkSize.value;
	ThreadPool::DefaultSchedule = (ThreadPool::ScheduleType)ScheduleType.value;
	ThreadPool::Init( (ThreadPool::ParallelType)ParallelType.value , Threads.value );

	if( !In.set )
	{
		ShowUsage( argv[0] );
		return 0;
	}

	if( !BaseDepth.set ) BaseDepth.value = FullDepth.value;
	if( !SolveDepth.set ) SolveDepth.value = Depth.value;

	if( BaseDepth.value>FullDepth.value )
	{
		if( BaseDepth.set ) WARN( "Base depth must be smaller than full depth: " , BaseDepth.value , " <= " , FullDepth.value );
		BaseDepth.value = FullDepth.value;
	}
	if( SolveDepth.value>Depth.value )
	{
		WARN( "Solution depth cannot exceed system depth: " , SolveDepth.value , " <= " , Depth.value );
		SolveDepth.value = Depth.value;
	}
	if( !KernelDepth.set ) KernelDepth.value = Depth.value-2;
	if( KernelDepth.value>Depth.value )
	{
		WARN( "Kernel depth should not exceed depth: " , KernelDepth.name , " <= " , KernelDepth.value );
		KernelDepth.value = Depth.value;
	}

	if( !EnvelopeDepth.set ) EnvelopeDepth.value = BaseDepth.value;
	if( EnvelopeDepth.value>Depth.value )
	{
		WARN( EnvelopeDepth.name , " can't be greater than " , Depth.name , ": " , EnvelopeDepth.value , " <= " , Depth.value );
		EnvelopeDepth.value = Depth.value;
	}
	if( EnvelopeDepth.value<BaseDepth.value )
	{
		WARN( EnvelopeDepth.name , " can't be less than " , BaseDepth.name , ": " , EnvelopeDepth.value , " >= " , BaseDepth.value );
		EnvelopeDepth.value = BaseDepth.value;
	}

#ifdef USE_DOUBLE
	typedef double Real;
#else // !USE_DOUBLE
	typedef float  Real;
#endif // USE_DOUBLE

#ifdef FAST_COMPILE
	static const int Degree = Reconstructor::Poisson::DefaultFEMDegree;
	static const BoundaryType BType = Reconstructor::Poisson::DefaultFEMBoundary;
	static const unsigned int Dim = DEFAULT_DIMENSION;
	static const unsigned int FEMSig = FEMDegreeAndBType< Degree , BType >::Signature;
	WARN( "Compiled for degree-" , Degree , ", boundary-" , BoundaryNames[ BType ] , ", " , sizeof(Real)==4 ? "single" : "double" , "-precision _only_" );
	if( !PointWeight.set ) PointWeight.value = Reconstructor::Poisson::WeightMultiplier*Degree;
	char *ext = GetFileExtension( In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties );
		if( !factory.plyValidReadProperties<0>( readFlags ) ) ERROR_OUT( "Ply file does not contain positions" );
		if( !factory.plyValidReadProperties<1>( readFlags ) ) ERROR_OUT( "Ply file does not contain normals" );
		delete[] readFlags;

		if( unprocessedProperties.size() ) Execute< Real , Dim , FEMSig >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< Real , Dim , FEMSig >( VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( Colors.set ) Execute< Real , Dim , FEMSig >( VertexFactory::RGBColorFactory< Real >() );
		else             Execute< Real , Dim , FEMSig >( VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#else // !FAST_COMPILE
	if( !PointWeight.set ) PointWeight.value = Reconstructor::Poisson::WeightMultiplier * Degree.value;
	char *ext = GetFileExtension( In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties );
		if( !factory.plyValidReadProperties<0>( readFlags ) ) ERROR_OUT( "Ply file does not contain positions" );
		if( !factory.plyValidReadProperties<1>( readFlags ) ) ERROR_OUT( "Ply file does not contain normals" );
		delete[] readFlags;

		if( unprocessedProperties.size() ) Execute< DEFAULT_DIMENSION , Real >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< DEFAULT_DIMENSION , Real >( VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( Colors.set ) Execute< DEFAULT_DIMENSION , Real >( VertexFactory::RGBColorFactory< Real >() );
		else             Execute< DEFAULT_DIMENSION , Real >( VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#endif // FAST_COMPILE
	if( Performance.set )
	{
		printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
		printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
	}

	ThreadPool::Terminate();
	return EXIT_SUCCESS;
}
