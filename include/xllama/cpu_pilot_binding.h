#pragma once
#include "xllama/model_write.h"
#include "windows_held_reader.h"

namespace xllama::cpu_pilot {
struct Binding {
    // Reverse destruction: FILE/Reader closures die before the writer permit.
    ModelWriteGate::Permit writer;
    ua50::uwp::WindowsInputCandidate original;
    explicit Binding(ModelWriteGate::Permit p):writer(std::move(p)){}
};
struct BoundControl final:ua50::uwp::WindowsInputControl {
    std::shared_ptr<Binding> binding;
    explicit BoundControl(std::shared_ptr<Binding> p):binding(std::move(p)){}
    void request_stop() noexcept override {binding->original.control->request_stop();}
    void drain_to_terminal() noexcept override {binding->original.control->drain_to_terminal();}
};
struct BoundReader final:ua50::uwp::WindowsReader {
    std::shared_ptr<Binding> binding;
    std::shared_ptr<ua50::uwp::WindowsReader> inner;
    BoundReader(std::shared_ptr<Binding> b,std::shared_ptr<ua50::uwp::WindowsReader> r)
        :binding(std::move(b)),inner(std::move(r)) {
        if(!inner)throw std::runtime_error("pilot Reader factory returned null");
    }
    size_t read(uint64_t offset,unsigned char* p,size_t n) override{return inner->read(offset,p,n);}
    void request_stop() noexcept override{inner->request_stop();}
    void drain_to_terminal() noexcept override{inner->drain_to_terminal();}
    ua50::uwp::Ledger ledger_after_drain() const override{return inner->ledger_after_drain();}
    ua50::uwp::Cause cause_after_drain() const override{return inner->cause_after_drain();}
};
template<class Source> ua50::uwp::WindowsInputCandidate bind_writer(ModelWriteGate::Permit permit,Source&& source) {
    if(!permit.owns_lock())throw std::invalid_argument("pilot writer unavailable");
    // Establish ordered ownership BEFORE invoking the source/hash factory, including unwind.
    auto b=std::make_shared<Binding>(std::move(permit));
    b->original=source();
    const auto& input=b->original;
    if(!input.file || !input.file_anchor || !input.admission_anchor ||
       !input.control || !input.validate_binding || !input.make_reader)
        throw std::invalid_argument("pilot writer/input binding incomplete");
    auto c=b->original;
    c.file_anchor=b;c.admission_anchor=b;
    c.validate_binding=[b](FILE* f){b->original.validate_binding(f);};
    c.make_reader=[b](size_t n)->std::shared_ptr<ua50::uwp::WindowsReader>{
        return std::make_shared<BoundReader>(b,b->original.make_reader(n));
    };
    c.control=std::make_shared<BoundControl>(b);
    return c;
}
} // namespace xllama::cpu_pilot
