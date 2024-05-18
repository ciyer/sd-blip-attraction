#![allow(non_snake_case)]

use crate::{image_and_desc_for_prompt, AttractionStatus, FixedPointKind, SdBlipResponse};
use dioxus::prelude::*;
use std::collections::HashMap;

#[component]
pub(crate) fn SdBlipAttractionHeader(max_depth: i32) -> Element {
    rsx! {
        div {
            class: "mb-4",
            div {
                class: "row",
                div {
                    class: "col-lg-10 col-xl-8",
                    h1 { "Stable Diffusion / BLIP Attraction" }
                    div {
                        class: "mb-2",
                        i { "SD/BLIP Attraction " },
                        b {"starts"}
                        span { " by generating an image from a prompt using " }
                        b { "Stable Diffusion"}
                        span {". The result is feed "}
                        b {"into BLIP"}
                        span { " to produce a description. The description becomes a "}
                        b{"new prompt"}
                        span {" to SD." }
                    }
                    div {
                        span { "This "}
                        b {"repeats"}
                        span { " until the BLIP description matches the SD prompt, or the maximum number of
                                iterations ({max_depth}) is reached."
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn SdBlipAttractionInput(
    prompt: Signal<String>,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    status: Signal<AttractionStatus>,
) -> Element {
    let mut input = use_signal(|| String::from("a cosmonaut on a horse (hd, realistic, high-def)"));
    let mut disabled = use_signal(|| (*status.read()) == AttractionStatus::Iterating);
    let _my_updater = use_effect(move || {
        disabled.set((*status.read()) == AttractionStatus::Iterating);
    });
    log::info!("status: {:?}", status.read());
    rsx! {
        form {
            onsubmit: move |e| async move {
                e.stop_propagation();
                prompt.set(input.to_string());
                // Add a placeholder for the response
                let mut r_map = HashMap::<i32, SdBlipResponse>::new();
                r_map.insert(0, SdBlipResponse {
                    prompt: prompt.to_string(),
                    image_base64: String::from(""),
                    description: String::from(""),
                });
                response_map.set(r_map.clone());
                status.set(AttractionStatus::Iterating);
                disabled.set(true);
            },
            div {
                class: "mb-3",
                label {
                    class: "form-label",
                    r#for: "prompt",
                    "Prompt"
                },
                input {
                    class: "form-control",
                    id: "prompt",
                    placeholder: "Enter a prompt to generate an image...",
                    r#type: "text",
                    value: input,
                    required: true,
                    disabled: disabled,
                    oninput: move |e| {
                        input.set(e.value());
                    }
                }
                div {
                    class: "invalid-feedback",
                    "Please provide a prompt."
                }
            }
            div {
                class: "d-flex flex-row-reverse",
                button {
                    class: "btn btn-primary btn-sm me-2",
                    r#type: "submit",
                    "Start"
                }
                button {
                    class: "btn btn-outline-secondary btn-sm me-2",
                    r#type: "button",
                    onclick: move |e| {
                        e.stop_propagation();
                        status.set(AttractionStatus::Interrupted);
                    },
                    "Stop"
                }
            }
        }
    }
}

#[component]
pub(crate) fn SdBlipAttractionResponse(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    status: Signal<AttractionStatus>,
) -> Element {
    let responses_map = response_map.read();
    let response = responses_map.get(&index);
    match response {
        Some(response) => {
            rsx! {
                 div {
                    class: "mb-3",
                    div {
                        class: "mt-4 text-muted fst-italic",
                        "{index+1:02}: {response.prompt}"
                    },
                    if response.description.len() > 0 {
                        SdBlipAttractionResponseReceived {index: index, response_map: response_map, description: response.description.clone(), image: response.image_base64.clone()}
                    } else {
                        SdBlipAttractionResponseGenerating {index: index, response_map: response_map, prompt: response.prompt.clone(), status: status}
                    },
                }
            }
        }
        _ => None,
    }
}

#[component]
fn SdBlipAttractionResponseGenerating(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    prompt: String,
    status: Signal<AttractionStatus>,
) -> Element {
    let _my_resource = use_resource(use_reactive(&prompt, move |prompt| async move {
        if *status.read() != AttractionStatus::Iterating {
            return;
        }
        if let Ok(response) = image_and_desc_for_prompt(prompt).await {
            let next_prompt = response.description.clone();
            let mut r_map = response_map.read().clone();
            r_map.insert(index, response);
            let have_fixed_point = HaveFixedPoint(index, response_map, &next_prompt);
            if have_fixed_point != FixedPointKind::None {
                // Set to idle
                status.set(AttractionStatus::Idle);
            } else {
                // Kick off the next generation
                r_map.insert(
                    index + 1,
                    SdBlipResponse {
                        prompt: next_prompt,
                        image_base64: String::from(""),
                        description: String::from(""),
                    },
                );
            }
            response_map.set(r_map.clone());
        }
    }));
    rsx! {
        div {
            class: "d-flex",
            div {
                class: "spinner-border text-primary",
                role: "status",
                span {
                    class: "visually-hidden",
                    "Generating..."
                }
            }
            div {
                class: "text-secondary ms-2",
                "Generating (please be patient, this takes some time)..."
            }
        }
    }
}

fn HaveFixedPoint(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    description: &String,
) -> FixedPointKind {
    let responses_map = response_map.read();
    let map_len = responses_map.len();
    let comp_desc = description.to_lowercase();
    if index < 1 {
        return FixedPointKind::None;
    };
    if index >= map_len as i32 {
        return FixedPointKind::None;
    }
    for i in 0..(index + 1) as i32 {
        let response = responses_map.get(&i);
        match response {
            Some(response) => {
                if response.prompt.to_lowercase() == comp_desc {
                    if index == i {
                        return FixedPointKind::FixedPoint;
                    } else {
                        return FixedPointKind::Loop;
                    }
                }
            }
            _ => (),
        }
    }
    return FixedPointKind::None;
}

#[component]
fn SdBlipAttractionResponseReceived(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    description: String,
    image: String,
) -> Element {
    let have_fixed_point = HaveFixedPoint(index, response_map, &description);
    rsx! {
        img { src: "data:image/png;base64,{image}", class: "img-fluid" }
        p { "{description}"}
        // Could not figure out how to get this to work with match
        if  have_fixed_point == FixedPointKind::FixedPoint {
            h3 {
                "Fixed point found!"
            }
        }
        else if have_fixed_point == FixedPointKind::Loop {
            h3 {
                "Loop found!"
            }
        }
    }
}

#[component]
pub(crate) fn SdBlipAttractionLastResponse(
    index: i32,
    max_depth: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    status: Signal<AttractionStatus>,
) -> Element {
    let responses_map = response_map.read();
    let response = responses_map.get(&index);
    match response {
        Some(response) => {
            rsx! {
                 div {
                    class: "mb-3",
                    div {
                        class: "mt-4 text-muted fst-italic",
                        "{index+1:02}: {response.prompt}"
                    },
                    if response.description.len() > 0 {
                        SdBlipAttractionLastResponseReceived { index: index, response_map: response_map, description: response.description.clone(), image: response.image_base64.clone() }
                    } else {
                        SdBlipAttractionLastResponseGenerating { index: index, response_map: response_map, prompt: response.prompt.clone(), status: status}
                    },
                }
            }
        }
        _ => None,
    }
}

#[component]
fn SdBlipAttractionLastResponseGenerating(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    prompt: String,
    status: Signal<AttractionStatus>,
) -> Element {
    let _my_resource = use_resource(use_reactive(&prompt, move |prompt| async move {
        if let Ok(response) = image_and_desc_for_prompt(prompt).await {
            let mut r_map = response_map.read().clone();
            r_map.insert(index, response);
            response_map.set(r_map.clone());
            // Set to idle
            status.set(AttractionStatus::Idle);
        }
    }));
    rsx! {
        div {
            class: "d-flex",
            div {
                class: "spinner-border text-primary",
                role: "status",
                span {
                    class: "visually-hidden",
                    "Generating..."
                }
            }
            div {
                class: "text-secondary ms-2",
                "Generating (please be patient, this takes some time)..."
            }
        }
    }
}

#[component]
fn SdBlipAttractionLastResponseReceived(
    index: i32,
    response_map: Signal<HashMap<i32, SdBlipResponse>>,
    description: String,
    image: String,
) -> Element {
    let have_fixed_point = HaveFixedPoint(index, response_map, &description);
    rsx! {
        img { src: "data:image/png;base64,{image}", class: "img-fluid" }
        p { "{description}"}
        // Could not figure out how to get this to work with match
        if  have_fixed_point == FixedPointKind::FixedPoint {
            h3 {
                "Fixed point found!"
            }
        }
        else if have_fixed_point == FixedPointKind::Loop {
            h3 {
                "Loop found!"
            }
        }
        else {
            h3 {
                "Iteration limit reached."
            }
        }
    }
}
