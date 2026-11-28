// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: chat history and feedback. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

void MainPageController::AppendFeedbackControls(
    winrt::Windows::UI::Xaml::Documents::Paragraph const& paragraph, size_t assistant_index) {
    using namespace winrt::Windows::UI::Xaml::Controls;
    using namespace winrt::Windows::UI::Xaml::Documents;

    if (assistant_index >= m_current.messages.size())
        return;
    const auto& msg = m_current.messages[assistant_index];
    if (msg.role != xllama::ui::MessageRole::Assistant || msg.partial)
        return;

    paragraph.Inlines().Append(LineBreak());
    if (!msg.feedback_label.empty()) {
        Run recorded;
        recorded.Text(L"Feedback recorded: " + ::xllama::utf8_to_wstring(msg.feedback_label));
        recorded.FontSize(13);
        paragraph.Inlines().Append(recorded);
        return;
    }

    StackPanel actions;
    actions.Orientation(Orientation::Horizontal);
    actions.Spacing(8);

    auto weak_self = weak_from_this();
    auto add_button = [&](const wchar_t* text, const std::string& label) {
        Button button;
        button.Content(winrt::box_value(text));
        button.MinWidth(96);
        button.Click(
            [weak_self, assistant_index, label](IInspectable const&, RoutedEventArgs const&) {
                if (auto self = weak_self.lock()) {
                    std::string err;
                    if (self->SubmitFeedback(assistant_index, label, {}, &err)) {
                        self->SetStatus(L"Feedback saved", StatusKind::Success);
                        self->RenderConversation();
                    } else {
                        self->SetStatus(L"Feedback failed: " + ::xllama::utf8_to_wstring(err),
                                        StatusKind::Error);
                    }
                }
            });
        actions.Children().Append(button);
    };
    add_button(L"Like", "like");
    add_button(L"Dislike", "dislike");

    Button correct;
    correct.Content(winrt::box_value(L"Correct"));
    correct.MinWidth(96);
    correct.Click([weak_self, assistant_index](IInspectable const&, RoutedEventArgs const&) {
        if (auto self = weak_self.lock())
            self->ShowCorrectionDialog(assistant_index);
    });
    actions.Children().Append(correct);

    InlineUIContainer container;
    container.Child(actions);
    paragraph.Inlines().Append(container);
}

bool MainPageController::SubmitFeedback(size_t assistant_index, const std::string& label,
                                        const std::string& preferred_assistant, std::string* err) {
    auto fail = [&](const std::string& message) {
        if (err)
            *err = message;
        return false;
    };
    if (!::xllama::preference_label_valid(label))
        return fail("invalid label");
    if (assistant_index >= m_current.messages.size())
        return fail("response no longer exists");
    auto& assistant = m_current.messages[assistant_index];
    if (assistant.role != xllama::ui::MessageRole::Assistant || assistant.partial)
        return fail("response is not rateable");
    if (!assistant.feedback_label.empty())
        return fail("response already rated");
    if (label == "correction" &&
        preferred_assistant.find_first_not_of(" \t\r\n") == std::string::npos)
        return fail("correction text is required");

    const xllama::ui::ChatMessage* user = nullptr;
    for (size_t i = assistant_index; i-- > 0;) {
        if (m_current.messages[i].role == xllama::ui::MessageRole::User) {
            user = &m_current.messages[i];
            break;
        }
    }
    if (!user)
        return fail("preceding user message not found");

    const std::vector<std::pair<std::string, std::string>> messages = {
        {"user", user->content}, {"assistant", assistant.content}};
    const std::string line =
        ::xllama::format_preference_sample_jsonl(label, messages, preferred_assistant);
    if (line.empty())
        return fail("could not format preference sample");

    CreateDirectoryW(local_wpath(L"training").c_str(), nullptr);
    const std::string samples_path =
        ::xllama::wstring_to_utf8(local_wpath(L"training\\samples.jsonl"));
    if (!::xllama::append_preference_sample_file(samples_path, line))
        return fail("could not append training/samples.jsonl");

    assistant.feedback_label = label;
    if (!SaveCurrentConversation())
        return fail("rating recorded but chat history could not be saved");
    log_output("[xllama] feedback label=" + label + " appended preference sample\n");
    return true;
}

winrt::fire_and_forget MainPageController::ShowCorrectionDialog(size_t assistant_index) {
    using namespace winrt::Windows::UI::Xaml::Controls;
    auto self = shared_from_this();
    if (assistant_index >= m_current.messages.size() ||
        !m_current.messages[assistant_index].feedback_label.empty())
        co_return;

    TextBox correction;
    no_keyboard_on_programmatic_focus(correction);
    correction.Header(winrt::box_value(L"Preferred answer"));
    correction.AcceptsReturn(true);
    correction.TextWrapping(TextWrapping::Wrap);
    correction.MinHeight(140);
    correction.IsFocusEngagementEnabled(true);

    ContentDialog dialog;
    dialog.Title(winrt::box_value(L"Correct this response"));
    dialog.Content(correction);
    dialog.PrimaryButtonText(L"Save correction");
    dialog.CloseButtonText(L"Cancel");
    dialog.XamlRoot(m_root.XamlRoot());
    ApTrackDialog(dialog);

    if (co_await dialog.ShowAsync() != ContentDialogResult::Primary)
        co_return;
    const std::string preferred =
        ::xllama::wstring_to_utf8(std::wstring(correction.Text().c_str()));
    std::string err;
    if (self->SubmitFeedback(assistant_index, "correction", preferred, &err)) {
        self->SetStatus(L"Correction saved", StatusKind::Success);
        self->RenderConversation();
    } else {
        self->SetStatus(L"Correction failed: " + ::xllama::utf8_to_wstring(err), StatusKind::Error);
    }
}

bool MainPageController::SaveCurrentConversation(bool partial) {
    if (m_current.id.empty())
        return true;
    // Mark last assistant message as partial if needed
    if (partial && !m_current.messages.empty() &&
        m_current.messages.back().role == xllama::ui::MessageRole::Assistant) {
        m_current.messages.back().partial = true;
    }
    if (m_history.Save(m_current))
        return true;
    SetStatus(L"Could not save chat history", StatusKind::Error);
    return false;
}

void MainPageController::NewChat() {
    if (m_is_running.load())
        return; // don't allow while running
    if (!SaveCurrentConversation())
        return;
    SaveKvSnapshotAsync();  // #170b: before m_kv_valid/m_active_model are cleared
    m_kv_valid = false;     // new conversation → discard reused KV
    m_active_model.clear(); // re-decide EP routing for the new conversation
    m_current = xllama::ui::Conversation{};
    m_current.id = xllama::ui::ChatHistory::NewId();
    m_outputBody.Blocks().Clear();
    m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_promptInput.Text(L"");
    m_metricsText.Text(L"");
    SetStatus(L"New conversation");
}

void MainPageController::RenderConversation() {
    using namespace winrt::Windows::UI::Xaml::Documents;
    m_outputBody.Blocks().Clear();
    // One format resolution for the whole render, not one per message (#174) —
    // chat_format_for builds ~10 strings per call and the format only depends
    // on m_model_filename, which cannot change mid-loop.
    const xllama::ChatFormat fmt = chat_format();
    for (size_t message_index = 0; message_index < m_current.messages.size(); ++message_index) {
        const auto& msg = m_current.messages[message_index];
        if (msg.role == xllama::ui::MessageRole::System)
            continue;
        Paragraph p;
        const wchar_t* role_label =
            (msg.role == xllama::ui::MessageRole::User) ? L"You: " : L"Assistant: ";
        Run label;
        label.Text(role_label);
        label.FontWeight(winrt::Windows::UI::Text::FontWeights::Bold());
        p.Inlines().Append(label);
        Run content;
        content.Text(::xllama::utf8_to_wstring(fmt.postprocess_output(msg.content)));
        if (msg.partial)
            content.Text(content.Text() + L" [incomplete]");
        p.Inlines().Append(content);
        if (msg.role == xllama::ui::MessageRole::Assistant)
            AppendFeedbackControls(p, message_index);
        m_outputBody.Blocks().Append(p);
        m_outputBody.Blocks().Append(Paragraph()); // spacing
    }
    // Prepare fresh paragraph for next assistant turn
    m_currentParagraph = Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_outputScroll.UpdateLayout();
    m_outputScroll.ChangeView(nullptr, m_outputScroll.ScrollableHeight(), nullptr);
}

void MainPageController::FinalizeStreamedTurn(const std::string& output_text) {
    using namespace winrt::Windows::UI::Xaml::Documents;
    // The streamed paragraph holds the raw pieces; swap in the postprocessed
    // text (the caller already ran postprocess_output) and attach the
    // feedback controls. The rest of the conversation tree is untouched —
    // the previous shape rebuilt every Paragraph/Run of the conversation
    // after each turn, discarding what streaming had just appended.
    if (!m_currentParagraph) {
        RenderConversation();
        return;
    }
    m_currentParagraph.Inlines().Clear();
    Run label;
    label.Text(L"Assistant: ");
    label.FontWeight(winrt::Windows::UI::Text::FontWeights::Bold());
    m_currentParagraph.Inlines().Append(label);
    Run content;
    content.Text(::xllama::utf8_to_wstring(output_text));
    m_currentParagraph.Inlines().Append(content);
    // The assistant message was just pushed onto m_current.messages.
    AppendFeedbackControls(m_currentParagraph, m_current.messages.size() - 1);
    m_outputBody.Blocks().Append(Paragraph()); // spacing
    m_currentParagraph = Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_outputScroll.UpdateLayout();
    m_outputScroll.ChangeView(nullptr, m_outputScroll.ScrollableHeight(), nullptr);
}

void MainPageController::LoadConversation(const std::string& id) {
    if (!SaveCurrentConversation())
        return;
    auto loaded = m_history.Load(id);
    if (loaded.id.empty()) {
        SetStatus(L"Could not read the selected conversation", StatusKind::Error);
        return;
    }
    SaveKvSnapshotAsync();  // #170b: before m_kv_valid/m_active_model are cleared
    m_kv_valid = false;     // switching conversations → the reused KV no longer applies
    m_active_model.clear(); // re-decide EP routing for the loaded conversation
    m_current = std::move(loaded);
    RenderConversation();
    SetStatus(L"Conversation loaded");
}

// Format unix timestamp as relative string ("today HH:MM", "yesterday HH:MM", "DD Mon HH:MM")
static std::wstring FormatRelativeTs(int64_t unix_ts) {
    if (unix_ts <= 0)
        return L"";
    std::time_t now = std::time(nullptr);
    std::time_t ts = static_cast<std::time_t>(unix_ts);
    std::tm now_tm{}, ts_tm{};
    #ifdef _WIN32
    localtime_s(&now_tm, &now);
    localtime_s(&ts_tm, &ts);
    #else
    localtime_r(&now, &now_tm);
    localtime_r(&ts, &ts_tm);
    #endif
    wchar_t buf[64];
    if (now_tm.tm_year == ts_tm.tm_year && now_tm.tm_yday == ts_tm.tm_yday)
        swprintf_s(buf, L"today %02d:%02d", ts_tm.tm_hour, ts_tm.tm_min);
    else if (now_tm.tm_year == ts_tm.tm_year && now_tm.tm_yday - ts_tm.tm_yday == 1)
        swprintf_s(buf, L"yesterday %02d:%02d", ts_tm.tm_hour, ts_tm.tm_min);
    else {
        static const wchar_t* months[] = {L"Jan", L"Feb", L"Mar", L"Apr", L"May", L"Jun",
                                          L"Jul", L"Aug", L"Sep", L"Oct", L"Nov", L"Dec"};
        swprintf_s(buf, L"%d %s %02d:%02d", ts_tm.tm_mday, months[ts_tm.tm_mon], ts_tm.tm_hour,
                   ts_tm.tm_min);
    }
    return buf;
}

winrt::fire_and_forget MainPageController::ShowHistory() {
    auto self = shared_from_this();
    if (m_is_running.load())
        co_return;

    if (!m_history.LoadIndex()) {
        SetStatus(L"Could not read conversation history", StatusKind::Error);
        co_return;
    }
    const auto& index = m_history.Index();

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Conversation History"));
    dlg.XamlRoot(m_root.XamlRoot());
    ApTrackDialog(dlg);

    if (index.empty()) {
        winrt::Windows::UI::Xaml::Controls::TextBlock empty_tb;
        empty_tb.Text(L"No conversations yet — start chatting to see history here.");
        empty_tb.FontSize(16);
        empty_tb.Opacity(0.6);
        empty_tb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        dlg.Content(empty_tb);
        dlg.CloseButtonText(L"Close");
        co_await dlg.ShowAsync();
        co_return;
    }

    // Build a ListView — each row: [title TextBlock | ✕ Delete button]
    winrt::Windows::UI::Xaml::Controls::ListView lv;
    lv.SelectionMode(winrt::Windows::UI::Xaml::Controls::ListViewSelectionMode::Single);
    lv.Height(400);
    // A ContentDialog caps its content near 548 effective px: the old 580 px list
    // with wrapping titles pushed the delete button past the dialog edge (seen
    // on-console). Rows are a fixed-width grid: title + details | delete.
    lv.Width(500);
    std::string current_id = m_current.id;

    // Shared state: when a Delete button fires it sets this and hides the dialog
    auto delete_pending = std::make_shared<std::string>();

    for (const auto& meta : index) {
        winrt::Windows::UI::Xaml::Controls::Grid row;
        row.Width(456);
        winrt::Windows::UI::Xaml::Controls::ColumnDefinition text_col;
        text_col.Width(winrt::Windows::UI::Xaml::GridLengthHelper::FromValueAndType(
            1, winrt::Windows::UI::Xaml::GridUnitType::Star));
        winrt::Windows::UI::Xaml::Controls::ColumnDefinition button_col;
        button_col.Width(winrt::Windows::UI::Xaml::GridLengthHelper::Auto());
        row.ColumnDefinitions().Append(text_col);
        row.ColumnDefinitions().Append(button_col);

        winrt::Windows::UI::Xaml::Controls::StackPanel text;
        text.VerticalAlignment(winrt::Windows::UI::Xaml::VerticalAlignment::Center);
        winrt::Windows::UI::Xaml::Controls::TextBlock tb;
        tb.FontSize(16);
        tb.TextTrimming(winrt::Windows::UI::Xaml::TextTrimming::CharacterEllipsis);
        std::wstring prefix = (meta.id == current_id) ? L"\u25CF " : L"";
        tb.Text(prefix + ::xllama::utf8_to_wstring(meta.title));
        winrt::Windows::UI::Xaml::Controls::TextBlock details;
        details.FontSize(14);
        details.Opacity(0.85);
        details.Text(std::to_wstring(meta.n_messages) +
                     (meta.n_messages == 1 ? L" message" : L" messages") + L"  \u2022  " +
                     FormatRelativeTs(meta.last_modified));
        text.Children().Append(tb);
        text.Children().Append(details);

        winrt::Windows::UI::Xaml::Controls::Button del_btn;
        del_btn.Content(winrt::box_value(L"✕"));
        del_btn.Width(48);
        del_btn.VerticalAlignment(winrt::Windows::UI::Xaml::VerticalAlignment::Center);
        del_btn.Margin(winrt::Windows::UI::Xaml::ThicknessHelper::FromLengths(8, 0, 0, 0));
        winrt::Windows::UI::Xaml::Controls::Grid::SetColumn(del_btn, 1);
        auto meta_id = meta.id;
        // Weak capture of dlg: a strong capture closed a refcount cycle
        // (dlg → Content → … → Button → Click → dlg) and leaked the whole
        // History dialog + rows per open (#219). Matches weak_self pattern above.
        auto weak_dlg = winrt::make_weak(dlg);
        del_btn.Click([delete_pending, meta_id, weak_dlg](
                          IInspectable const&, winrt::Windows::UI::Xaml::RoutedEventArgs const&) {
            *delete_pending = meta_id;
            if (auto d = weak_dlg.get())
                d.Hide();
        });

        row.Children().Append(text);
        row.Children().Append(del_btn);
        lv.Items().Append(row);
    }

    dlg.Content(lv);
    dlg.PrimaryButtonText(L"Open");
    dlg.SecondaryButtonText(L"Clear all");
    dlg.CloseButtonText(L"Cancel");

    auto result = co_await dlg.ShowAsync();

    // Per-item delete: a Delete button was pressed, hide closed the dialog
    if (!delete_pending->empty()) {
        std::string id_to_delete = *delete_pending;
        winrt::Windows::UI::Xaml::Controls::ContentDialog confirm;
        confirm.Title(winrt::box_value(L"Delete conversation?"));
        winrt::Windows::UI::Xaml::Controls::TextBlock ctb;
        ctb.Text(L"This will permanently delete this conversation.");
        ctb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        confirm.Content(ctb);
        confirm.PrimaryButtonText(L"Delete");
        confirm.CloseButtonText(L"Cancel");
        confirm.XamlRoot(m_root.XamlRoot());
        ApTrackDialog(confirm);
        auto cr = co_await confirm.ShowAsync();
        if (cr == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary) {
            bool was_current = (id_to_delete == self->m_current.id);
            self->m_history.Delete(id_to_delete);
            kv_store().erase(id_to_delete); // #170b: no orphan snapshot
            if (was_current) {
                // Drop the in-memory copy FIRST: NewChat starts by saving the
                // current conversation, which would write the just-deleted one
                // straight back to disk (and, with #170b, re-create its
                // snapshot).
                self->m_current = xllama::ui::Conversation{};
                self->NewChat();
            }
            self->SetStatus(L"Conversation deleted");
        }
        co_return;
    }

    if (result == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Secondary) {
        // Clear all conversations — confirm first
        winrt::Windows::UI::Xaml::Controls::ContentDialog confirm;
        confirm.Title(winrt::box_value(L"Clear all conversations?"));
        winrt::Windows::UI::Xaml::Controls::TextBlock ctb;
        ctb.Text(L"This will permanently delete all conversation history.");
        ctb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        confirm.Content(ctb);
        confirm.PrimaryButtonText(L"Delete all");
        confirm.CloseButtonText(L"Cancel");
        confirm.XamlRoot(m_root.XamlRoot());
        ApTrackDialog(confirm);
        auto cr = co_await confirm.ShowAsync();
        if (cr == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary) {
            self->m_history.Clear();
            // #170b: snapshots are per-conversation caches — clearing the
            // history must not leave their bytes on disk.
            kv_store().prune(/*max_files=*/0, /*max_bytes=*/0);
            // Same trap as the single delete: NewChat would save the current
            // conversation back into the history we just cleared.
            self->m_current = xllama::ui::Conversation{};
            self->NewChat();
            self->SetStatus(L"All conversations cleared");
        }
        co_return;
    }

    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    int sel = lv.SelectedIndex();
    if (sel < 0 || sel >= static_cast<int>(index.size()))
        co_return;

    self->LoadConversation(index[static_cast<size_t>(sel)].id);
}

} // namespace xllama

#endif // XLLAMA_UWP
