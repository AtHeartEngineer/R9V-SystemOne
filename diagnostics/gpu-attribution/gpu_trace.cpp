// SPDX-License-Identifier: Apache-2.0
// Incident-image SDK callbacks. No synchronization, counter collection, or HIP calls.
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <unistd.h>
#include <time.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <sys/stat.h>

namespace {
std::mutex lock;
uint64_t sequence = 0;
rocprofiler_context_id_t context{};
std::unordered_map<uint64_t, std::string> names;
std::unordered_map<uint64_t, uint64_t> object_ids;
uint64_t stamp(clockid_t clock) {
    timespec t{}; clock_gettime(clock, &t);
    return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
std::string quote(const char* p) {
    std::string out="\"";
    if(p) for(size_t n=0; p[n] && n<1400; ++n) {
        unsigned char c=p[n];
        if(c=='"' || c=='\\') { out+='\\'; out+=c; }
        else if(c>=32 && c<127) out+=c;
        else { char b[7]; snprintf(b,sizeof(b),"\\u%04x",c); out+=b; }
    }
    return out+'"';
}
void emit(const std::string& fields) {
    // Caller holds lock. One bounded write avoids interleaving on the container pipe.
    std::ostringstream s;
    s << "R9V_GPU {\"pid\":" << getpid() << ",\"seq\":" << ++sequence
      << ",\"mono_ns\":" << stamp(CLOCK_MONOTONIC)
      << ",\"wall_ns\":" << stamp(CLOCK_REALTIME) << ',' << fields << "}\n";
    auto line=s.str();
    if(line.size()>4000) { line="R9V_GPU {\"error\":\"oversize record\"}\n"; }
    ssize_t n;
    do { n=write(STDERR_FILENO,line.data(),line.size()); } while(n<0 && errno==EINTR);
    // A sequence gap or explicit error fails evidence admission; never silently claim completeness.
    if(n!=ssize_t(line.size())) {
        constexpr char failure[]="R9V_GPU {\"error\":\"trace write failed\"}\n";
        (void)write(STDERR_FILENO,failure,sizeof(failure)-1);
    }
}
void save_object(const rocprofiler_callback_tracing_code_object_load_data_t& p) {
    static uint64_t total = 0;
    int src = -1;
    uint64_t offset = 0, bytes = 0;
    if(p.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
        src = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
        offset = p.memory_base;
        bytes = p.memory_size;
    } else if(p.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) {
        src = dup(p.storage_file);
        struct stat st{};
        if(src >= 0 && fstat(src, &st) == 0) bytes = st.st_size;
    }
    std::string dir = "/capture/debug-" + std::to_string(getpid());
    std::string path = dir + "/object-" + std::to_string(p.code_object_id) + ".elf";
    if(src < 0 || !bytes || bytes > 256ULL*1024*1024 || total + bytes > 768ULL*1024*1024) {
        if(src >= 0) close(src);
        emit("\"error\":\"code object unavailable or over budget\",\"code_object\":" + std::to_string(p.code_object_id));
        return;
    }
    mkdir(dir.c_str(), 0700);
    std::string partial = path + ".partial";
    int dst = open(partial.c_str(), O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0600);
    bool ok = dst >= 0;
    char buf[65536];
    uint64_t done = 0;
    while(ok && done < bytes) {
        size_t count = std::min<uint64_t>(sizeof(buf), bytes-done);
        ssize_t n;
        do { n = pread(src, buf, count, offset+done); } while(n<0 && errno==EINTR);
        if(n<=0) { ok=false; break; }
        ssize_t wrote=0;
        while(wrote<n) {
            ssize_t part=write(dst,buf+wrote,n-wrote);
            if(part<0 && errno==EINTR) continue;
            if(part<=0) { ok=false; break; }
            wrote+=part;
        }
        done+=n;
    }
    close(src);
    if(dst>=0) { if(fsync(dst)!=0) ok=false; close(dst); }
    if(ok && rename(partial.c_str(),path.c_str())==0) {
        total += bytes;
        emit("\"type\":\"object_saved\",\"code_object\":"+std::to_string(p.code_object_id)+",\"bytes\":"+std::to_string(bytes)+",\"path\":"+quote(path.c_str()));
    } else emit("\"error\":\"code object copy failed\",\"code_object\":"+std::to_string(p.code_object_id));
}
void callback(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t*, void*) {
    std::lock_guard<std::mutex> guard(lock);
    std::ostringstream s;
    s << "\"kind\":" << r.kind << ",\"op\":" << r.operation
      << ",\"phase\":" << r.phase << ",\"tid\":" << r.thread_id
      << ",\"correlation\":" << r.correlation_id.internal;
    if(r.kind==ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) {
        if(r.operation==ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
            auto* p=static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(r.payload);
            if(r.phase==ROCPROFILER_CALLBACK_PHASE_LOAD) {
                names[p->kernel_id]=p->kernel_name;
                object_ids[p->kernel_id]=p->code_object_id;
            }
            s << ",\"type\":\"symbol\",\"kernel\":" << p->kernel_id
              << ",\"code_object\":" << p->code_object_id << ",\"name\":" << quote(p->kernel_name);
        } else if(r.operation==ROCPROFILER_CODE_OBJECT_LOAD) {
            auto* p=static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(r.payload);
            if(r.phase==ROCPROFILER_CALLBACK_PHASE_LOAD) save_object(*p);
            s << ",\"type\":\"code_object\",\"code_object\":" << p->code_object_id
              << ",\"agent\":" << p->agent_id.handle << ",\"base\":" << p->load_base
              << ",\"size\":" << p->load_size << ",\"uri\":" << quote(p->uri);
        } else return;
    } else if(r.kind==ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
        // ENQUEUE EXIT is not completion: only the COMPLETE operation is labelled complete.
        if(r.operation==ROCPROFILER_KERNEL_DISPATCH_ENQUEUE && r.phase==ROCPROFILER_CALLBACK_PHASE_EXIT) return;
        auto* p=static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(r.payload);
        auto& d=p->dispatch_info;
        auto it=names.find(d.kernel_id);
        s << ",\"type\":" << quote(r.operation==ROCPROFILER_KERNEL_DISPATCH_COMPLETE?"complete":"enqueue")
          << ",\"agent\":" << d.agent_id.handle << ",\"queue\":" << d.queue_id.handle
          << ",\"dispatch\":" << d.dispatch_id << ",\"kernel\":" << d.kernel_id
          << ",\"code_object\":" << object_ids[d.kernel_id]
          << ",\"name\":" << quote(it==names.end()?"UNKNOWN":it->second.c_str())
          << ",\"gpu_start\":" << p->start_timestamp << ",\"gpu_end\":" << p->end_timestamp
          << ",\"grid\":[" << d.grid_size.x << ',' << d.grid_size.y << ',' << d.grid_size.z << ']'
          << ",\"group\":[" << d.workgroup_size.x << ',' << d.workgroup_size.y << ',' << d.workgroup_size.z << ']';
    } else if(r.kind==ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
        auto* p=static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(r.payload);
        s << ",\"type\":\"copy\",\"bytes\":" << p->bytes
          << ",\"src_agent\":" << p->src_agent_id.handle << ",\"dst_agent\":" << p->dst_agent_id.handle
          << ",\"gpu_start\":" << p->start_timestamp << ",\"gpu_end\":" << p->end_timestamp;
    } else return;
    emit(s.str());
}
int init(rocprofiler_client_finalize_t,void*) {
    auto check=[](rocprofiler_status_t s) {
        if(s==ROCPROFILER_STATUS_SUCCESS) return true;
        std::lock_guard<std::mutex> guard(lock);
        emit("\"error\":\"SDK configuration failed\",\"status\":"+std::to_string(s)); return false;
    };
    if(!check(rocprofiler_create_context(&context))) return -1;
    for(auto kind : {ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                     ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH})
        if(!check(rocprofiler_configure_callback_tracing_service(context,kind,nullptr,0,callback,nullptr))) return -1;
    if(!check(rocprofiler_start_context(context))) return -1;
    std::lock_guard<std::mutex> guard(lock); emit("\"type\":\"ready\""); return 0;
}
void fini(void*) {
    std::lock_guard<std::mutex> guard(lock); emit("\"type\":\"finalize\"");
}
}
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version,const char*,uint32_t,rocprofiler_client_id_t* id) {
    id->name="r9v-durable-dispatch";
    static rocprofiler_tool_configure_result_t cfg{sizeof(cfg),init,fini,nullptr};
    std::lock_guard<std::mutex> guard(lock);
    emit("\"type\":\"configure\",\"sdk_version\":"+std::to_string(version));
    return &cfg;
}
