#if !defined(_WIN32)
#error "Windows implementation requires Windows/UWP SDK; portable tests cover only staged_reader_core.h"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <system_error>
#include "windows_held_reader.h"

namespace ua50::uwp {
namespace {
[[noreturn]] void winfail(const char* what) {
    throw std::system_error(static_cast<int>(GetLastError()),std::system_category(),what);
}
struct Handle {
    HANDLE value=INVALID_HANDLE_VALUE;
    Handle()=default;
    explicit Handle(HANDLE h):value(h){}
    Handle(Handle&& o) noexcept:value(o.release()){}
    Handle& operator=(Handle&& o) noexcept {
        if(this!=&o) {
            if(value!=INVALID_HANDLE_VALUE && value!=nullptr) CloseHandle(value);
            value=o.release();
        }
        return *this;
    }
    Handle(const Handle&)=delete;
    Handle& operator=(const Handle&)=delete;
    ~Handle(){ if(value!=INVALID_HANDLE_VALUE && value!=nullptr) CloseHandle(value); }
    HANDLE release() noexcept { auto h=value;value=INVALID_HANDLE_VALUE;return h; }
};
struct Identity {
    FILE_ID_INFO id{};
    FILE_STANDARD_INFO standard{};
    FILE_BASIC_INFO basic{};
};
bool inspect(HANDLE h,Identity& out) noexcept {
    FILE_ATTRIBUTE_TAG_INFO tag{};
    return GetFileType(h)==FILE_TYPE_DISK &&
        GetFileInformationByHandleEx(h,FileIdInfo,&out.id,sizeof(out.id)) &&
        GetFileInformationByHandleEx(h,FileStandardInfo,&out.standard,sizeof(out.standard)) &&
        GetFileInformationByHandleEx(h,FileBasicInfo,&out.basic,sizeof(out.basic)) &&
        GetFileInformationByHandleEx(h,FileAttributeTagInfo,&tag,sizeof(tag)) &&
        !(tag.FileAttributes&(FILE_ATTRIBUTE_REPARSE_POINT|FILE_ATTRIBUTE_DIRECTORY)) &&
        !out.standard.Directory && !out.standard.DeletePending &&
        out.standard.NumberOfLinks==1 && out.standard.EndOfFile.QuadPart>0;
}
bool same(const Identity& a,const Identity& b) noexcept {
    return a.id.VolumeSerialNumber==b.id.VolumeSerialNumber &&
        std::memcmp(a.id.FileId.Identifier,b.id.FileId.Identifier,sizeof(a.id.FileId.Identifier))==0 &&
        a.standard.EndOfFile.QuadPart==b.standard.EndOfFile.QuadPart &&
        a.basic.CreationTime.QuadPart==b.basic.CreationTime.QuadPart &&
        a.basic.LastWriteTime.QuadPart==b.basic.LastWriteTime.QuadPart &&
        a.basic.ChangeTime.QuadPart==b.basic.ChangeTime.QuadPart;
}
struct FileOwner final : WindowsInputControl {
    FILE* file=nullptr;
    Handle async_file;
    Identity identity{};
    std::mutex factory_mutex;
    bool reader_created=false;
    std::atomic<bool> stopping{false};
    std::weak_ptr<WindowsReader> reader;
    ~FileOwner(){ if(file) std::fclose(file); }
    bool stable() const noexcept {
        Identity a{},b{};
        auto sync=reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file)));
        return inspect(sync,a) && inspect(async_file.value,b) &&
            same(a,identity) && same(b,identity);
    }
    void request_stop() noexcept override {
        stopping.store(true,std::memory_order_release);
        std::shared_ptr<WindowsReader> live;
        { std::lock_guard<std::mutex> guard(factory_mutex);live=reader.lock(); }
        if(live) live->request_stop();
    }
    void drain_to_terminal() noexcept override {
        request_stop();
        std::shared_ptr<WindowsReader> live;
        { std::lock_guard<std::mutex> guard(factory_mutex);live=reader.lock(); }
        if(live) live->drain_to_terminal();
    }
};
class WindowsIo {
    std::shared_ptr<FileOwner> file_;
    Handle event_;
    OVERLAPPED operation_{};
    bool pending_=false;
public:
    explicit WindowsIo(std::shared_ptr<FileOwner> file):file_(std::move(file)),
        event_(CreateEventExW(nullptr,nullptr,CREATE_EVENT_MANUAL_RESET,EVENT_ALL_ACCESS)) {
        if(!event_.value) winfail("CreateEventExW");
    }
    WindowsIo(WindowsIo&&)=default;
    WindowsIo(const WindowsIo&)=delete;
    bool identity() const noexcept { return !pending_ && file_->stable(); }
    IoResult start(uint64_t offset,unsigned char* stage,size_t bytes) noexcept {
        if(pending_ || bytes>MAXDWORD) return {true,0,ERROR_INVALID_PARAMETER};
        if(!ResetEvent(event_.value)) return {true,0,GetLastError()};
        operation_={};operation_.Offset=static_cast<DWORD>(offset);
        operation_.OffsetHigh=static_cast<DWORD>(offset>>32);operation_.hEvent=event_.value;
        // For asynchronous handles, ReadFile's byte-count pointer is NULL.
        // Even immediate success is collected through the same completion path.
        if(ReadFile(file_->async_file.value,stage,static_cast<DWORD>(bytes),nullptr,&operation_)) {
            pending_=true;return {false,0,0};
        }
        const DWORD error=GetLastError();
        if(error!=ERROR_IO_PENDING) return {true,0,error};
        pending_=true;return {false,0,0};
    }
    void cancel() noexcept {
        // ERROR_NOT_FOUND can race completion; neither it nor success is a drain proof.
        if(pending_) (void)CancelIoEx(file_->async_file.value,&operation_);
    }
    IoResult poll(uint32_t milliseconds) noexcept {
        DWORD done=0;
        if(GetOverlappedResultEx(file_->async_file.value,&operation_,&done,milliseconds,FALSE)) {
            pending_=false;return {true,done,0};
        }
        const DWORD error=GetLastError();
        if(error==WAIT_TIMEOUT || error==ERROR_IO_INCOMPLETE || error==WAIT_IO_COMPLETION)
            return {false,0,0};
        // Only the kernel-signalled, private manual-reset event permits error
        // completion. Unexpected API/wait failure with no signal stays UNKNOWN.
        if(WaitForSingleObjectEx(event_.value,0,FALSE)==WAIT_OBJECT_0) {
            pending_=false;return {true,done,error};
        }
        return {false,0,error?error:ERROR_INVALID_DATA};
    }
};
class PlatformReader final : public WindowsReader {
    StagedReaderCore<WindowsIo> core_;
public:
    PlatformReader(std::shared_ptr<FileOwner> file,size_t stage,Clock::time_point deadline)
        :core_(WindowsIo(file),static_cast<uint64_t>(file->identity.standard.EndOfFile.QuadPart),stage,deadline){}
    size_t read(uint64_t offset,unsigned char* destination,size_t bytes) override {
        auto keep_alive=shared_from_this();
        return core_.read(offset,destination,bytes);
    }
    void request_stop() noexcept override { core_.request_stop(); }
    void drain_to_terminal() noexcept override { core_.drain_to_terminal(); }
    Ledger ledger_after_drain() const override { return core_.ledger_after_drain(); }
    Cause cause_after_drain() const override { return core_.cause_after_drain(); }
};
Handle open_file(const std::wstring& path,bool overlapped) {
    CREATEFILE2_EXTENDED_PARAMETERS p{};p.dwSize=sizeof(p);
    p.dwFileAttributes=FILE_ATTRIBUTE_NORMAL;
    p.dwFileFlags=FILE_FLAG_OPEN_REPARSE_POINT|(overlapped?FILE_FLAG_OVERLAPPED:0);
    Handle result(CreateFile2(path.c_str(),GENERIC_READ,FILE_SHARE_READ,OPEN_EXISTING,&p));
    if(result.value==INVALID_HANDLE_VALUE) winfail("CreateFile2 read/share-read");
    return result;
}
} // namespace
WindowsInputCandidate open_windows_input_candidate(
    const std::wstring& path,const ExpectedFile& expected,const ValidateHeldFile& validate,
    Clock::time_point deadline,size_t stage_ceiling) {
    if(path.empty() || !expected.bytes || expected.sha256.size()!=64 || !validate ||
       !stage_ceiling || stage_ceiling>4u*1024u*1024u || deadline<=Clock::now() ||
       !std::all_of(expected.sha256.begin(),expected.sha256.end(),[](char c){return
           (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
        throw std::invalid_argument("explicit Windows held-source admission required");
    auto owner=std::make_shared<FileOwner>();
    auto sync=open_file(path,false);
    owner->async_file=open_file(path,true);
    Identity second{};
    if(!inspect(sync.value,owner->identity) || !inspect(owner->async_file.value,second) ||
       !same(owner->identity,second) ||
       static_cast<uint64_t>(owner->identity.standard.EndOfFile.QuadPart)!=expected.bytes)
        throw std::runtime_error("held loader/Reader files do not match expected identity/size");
    const int fd=_open_osfhandle(reinterpret_cast<intptr_t>(sync.value),_O_RDONLY|_O_BINARY);
    if(fd==-1) throw std::runtime_error("_open_osfhandle");
    sync.release(); // descriptor now owns original synchronous HANDLE
    owner->file=_fdopen(fd,"rb");
    if(!owner->file) { _close(fd);throw std::runtime_error("_fdopen"); }
    // Hashing is supplied by the trusted admission caller, never pathname re-open.
    const auto actual=validate(owner->file,expected.bytes);
    if(actual.bytes_checked!=expected.bytes || actual.sha256!=expected.sha256 ||
       !owner->stable() || _fseeki64(owner->file,0,SEEK_SET)!=0 || Clock::now()>=deadline)
        throw std::runtime_error("whole held-file validation failed");
    std::ostringstream id;id<<std::hex<<owner->identity.id.VolumeSerialNumber<<':';
    for(auto v:owner->identity.id.FileId.Identifier)
        id<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(v);
    WindowsInputCandidate result;
    result.file=owner->file;result.file_anchor=owner;
    result.admission_anchor=owner;result.control=owner;
    result.admission={expected.bytes,expected.sha256,id.str()};
    result.validate_binding=[owner,deadline](FILE* file) {
        if(file!=owner->file || owner->stopping.load(std::memory_order_acquire) ||
           Clock::now()>=deadline || !owner->stable())
            throw std::runtime_error("Windows held binding invalid or expired");
    };
    result.make_reader=[owner,deadline,stage_ceiling](size_t required)->std::shared_ptr<WindowsReader> {
        std::lock_guard<std::mutex> guard(owner->factory_mutex);
        if(owner->reader_created || owner->stopping.load(std::memory_order_acquire) ||
           !required || required>stage_ceiling || !owner->stable())
            throw std::runtime_error("single bounded Reader admission");
        owner->reader_created=true;
        auto reader=std::make_shared<PlatformReader>(owner,required,deadline);
        owner->reader=reader;
        return reader;
    };
    return result;
}
} // namespace ua50::uwp
