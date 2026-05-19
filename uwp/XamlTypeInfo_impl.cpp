// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

// Provides implementations of XamlTypeInfoProvider private methods that
// MarkupCompilePass2 would normally generate. Because WMC9999 prevents Pass2
// from running for C++/WinRT UWP projects (crash in both SDK 22621 and 26100),
// these are written manually as no-op / minimal stubs.
//
// XamlTypeInfo.Impl.g.cpp (auto-compiled via GeneratedCodeFiles) declares
// XamlTypeInfoProvider with these private methods but does not define them.
// Returning null / empty is safe because our XAML uses no x:Bind or custom
// type mappings that require runtime type resolution.
#ifdef XLLAMA_UWP
    #include "XamlTypeInfo.xaml.g.h"
    #include "pch.h"
    #include <memory>
    #include <string>
    #include <vector>
    #include <winrt/Windows.UI.Xaml.Markup.h>

namespace winrt::xllama::implementation {

::winrt::Windows::UI::Xaml::Markup::IXamlType
XamlTypeInfoProvider::CreateXamlType(::winrt::hstring const&) {
    return nullptr;
}

::winrt::Windows::UI::Xaml::Markup::IXamlMember
XamlTypeInfoProvider::CreateXamlMember(::winrt::hstring const&) {
    return nullptr;
}

std::vector<::winrt::Windows::UI::Xaml::Markup::IXamlMetadataProvider> const&
XamlTypeInfoProvider::OtherProviders() {
    static std::vector<::winrt::Windows::UI::Xaml::Markup::IXamlMetadataProvider> s_empty;
    return s_empty;
}

} // namespace winrt::xllama::implementation
#endif // XLLAMA_UWP
