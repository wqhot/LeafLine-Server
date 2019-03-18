#include "pca.h"
#include<windows.h>

void exeMatlabDATA2SVM()
{
	STARTUPINFO si; 
	PROCESS_INFORMATION pi; 

	ZeroMemory( &si, sizeof(si) ); 
	si.cb = sizeof(si); 
	ZeroMemory( &pi, sizeof(pi) ); 
	if( !CreateProcess( "data2svm.exe",
		NULL,   // Command line.  There should be a space at the beginning
		NULL, // Process handle not inheritable. 
		NULL, // Thread handle not inheritable. 
		FALSE, // Set handle inheritance to FALSE. 
		0, // No creation flags. 
		NULL, // Use parent's environment block. 
		  NULL, // Use parent's starting directory. 
		&si, // Pointer to STARTUPINFO structure. 
		&pi ) )// Pointer to PROCESS_INFORMATION structure. 
	{ 
		printf( "CreateProcess failed (%d)./n", GetLastError() ); 
	} 
	// Wait until child process exits. 
	WaitForSingleObject( pi.hProcess, INFINITE ); 
	CloseHandle( pi.hProcess ); 
	CloseHandle( pi.hThread );
}
void exeSVMTrain()
{
	STARTUPINFO si; 
	PROCESS_INFORMATION pi; 

	ZeroMemory( &si, sizeof(si) ); 
	si.cb = sizeof(si); 
	ZeroMemory( &pi, sizeof(pi) ); 
	
	stringstream ss;
	float g,c;
	cout<<"g=";
	cin>>g;
	cout<<"c=";
	cin>>c;

	ss<<" -s 0 -t 2 -g "<<g<<" -c "<<c<<" data2svm.txt savedata\\svm_model";
	string str=ss.str();
	const char *aaa=str.data();
	if( !CreateProcess( "svm-train.exe",
		(LPSTR)str.c_str(),   // Command line.  There should be a space at the beginning
		NULL, // Process handle not inheritable. 
		NULL, // Thread handle not inheritable. 
		FALSE, // Set handle inheritance to FALSE. 
		0, // No creation flags. 
		NULL, // Use parent's environment block. 
		  NULL, // Use parent's starting directory. 
		&si, // Pointer to STARTUPINFO structure. 
		&pi ) )// Pointer to PROCESS_INFORMATION structure. 
	{ 
		printf( "CreateProcess failed (%d)./n", GetLastError() ); 
	} 
	// Wait until child process exits. 
	WaitForSingleObject( pi.hProcess, INFINITE ); 
	CloseHandle( pi.hProcess ); 
	CloseHandle( pi.hThread );
}
