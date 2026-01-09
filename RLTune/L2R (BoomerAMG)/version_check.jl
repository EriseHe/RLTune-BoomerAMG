using HYPRE
using Libdl

lib_path = HYPRE.LibHYPRE.HYPRE_jll.libHYPRE
lib_handle = HYPRE.LibHYPRE.HYPRE_jll.libHYPRE_handle

vptr = Ref{Ptr{Cchar}}()
ccall((:HYPRE_Version, lib_path), Cint, (Ref{Ptr{Cchar}},), vptr)
vstr = unsafe_string(vptr[])

major = Ref{Cint}()
minor = Ref{Cint}()
patch = Ref{Cint}()
single = Ref{Cint}()
ccall((:HYPRE_VersionNumber, lib_path), Cint,
      (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}),
      major, minor, patch, single)

println("version string: ", vstr)
println("version number: ", single[])
println("version: ", major[], ".", minor[], ".", patch[])
println("CumNnzAP set symbol: ", Libdl.dlsym_e(lib_handle, :HYPRE_BoomerAMGSetCumNnzAP))
println("CumNnzAP get symbol: ", Libdl.dlsym_e(lib_handle, :HYPRE_BoomerAMGGetCumNnzAP))
