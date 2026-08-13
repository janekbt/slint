// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Code generator for the Slint SC (safety-critical) runtime.

use crate::CompilerConfiguration;
use crate::embedded_resources::{EmbeddedResources, EmbeddedResourcesIdx, EmbeddedResourcesKind};
use crate::expression_tree::{BuiltinFunction, Callable, Expression, ImageReference, Unit};
use crate::langtype::{EnumerationValue, StructName, Type};
use crate::namedreference::NamedReference;
use crate::object_tree::{Document, ElementRc, PropertyVisibility};
use itertools::Either;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use typed_index_collections::TiSlice;

/// State threaded through expression compilation: the component whose code is
/// generated, and the document-wide image table.
struct Ctx<'a> {
    /// The root element of the component.
    root: &'a ElementRc,
    images: &'a ImageTable<'a>,
}

/// The document's image resources. An image referenced by a compiled
/// expression is emitted once as a `static`, shared by every reference.
struct ImageTable<'a> {
    resources: &'a TiSlice<EmbeddedResourcesIdx, EmbeddedResources>,
    /// The static's identifier and definition, by resource index, for the
    /// images referenced so far.
    emitted: RefCell<BTreeMap<usize, (Ident, TokenStream)>>,
}

impl ImageTable<'_> {
    /// The expression reading a compile-time decoded image, emitting the
    /// backing `static` on first use. The static holds the whole
    /// `slint_sc::Image` value, so the expression is just its name.
    fn image_expression(&self, resource_id: EmbeddedResourcesIdx) -> TokenStream {
        let mut emitted = self.emitted.borrow_mut();
        let idx = usize::from(resource_id);
        let (ident, _) = emitted.entry(idx).or_insert_with(|| {
            let EmbeddedResourcesKind::StaticPixels(image) = &self.resources[resource_id].kind
            else {
                unreachable!("the embedding pass only produces static pixels for Slint SC")
            };
            let ident = format_ident!("SLINT_SC_IMAGE_{idx}");
            let width = image.width() as usize;
            // The pixels are emitted packed into a single byte-string
            // literal, alpha first like the runtime's ARGB layout: one token
            // however large the image, where an array of per-pixel
            // constructor calls would multiply the source size and its
            // compile time.
            let bytes: Vec<u8> = image
                .pixels()
                .flat_map(|pixel| {
                    let [red, green, blue, alpha] = pixel.0;
                    [alpha, red, green, blue]
                })
                .collect();
            let data = proc_macro2::Literal::byte_string(&bytes);
            let definition = quote! {
                static #ident: slint_sc::Image =
                    slint_sc::Image::StaticArgb { argb: #data, width: #width };
            };
            (ident, definition)
        });
        let ident = ident.clone();
        quote!(#ident)
    }

    /// The definitions of the emitted statics, in resource order.
    fn statics(&self) -> TokenStream {
        self.emitted.borrow().values().map(|(_, definition)| definition.clone()).collect()
    }
}

/// Public entry point called from `generator::generate`.
pub fn generate(
    doc: &Document,
    _compiler_config: &CompilerConfiguration,
) -> std::io::Result<TokenStream> {
    let mut output = TokenStream::new();

    // The user-declared structs and enums, sorted so a struct's field types are
    // defined before the struct that uses them.
    for ty in doc.used_types.borrow().structs_and_enums.iter() {
        output.extend(generate_type_definition(ty));
    }

    let embedded_resources = doc.embedded_file_resources.borrow();
    let images = ImageTable { resources: &embedded_resources, emitted: Default::default() };

    for (export_name, export) in doc.exports.iter() {
        let Either::Left(component) = export else { continue };
        if component.is_global() {
            continue;
        }
        let root = &component.root_element;
        let ctx = Ctx { root, images: &images };
        let render_tree = emit_element(root, &ctx);
        let properties = declared_properties(&ctx);
        let name = format_ident!("{}", export_name.name.as_str());
        let has_fields = properties.iter().any(|p| p.field.is_some());
        let (struct_decl, new_body) = if has_fields {
            let fields = properties.iter().filter_map(|p| {
                let (field, ty) = (p.field.as_ref()?, &p.ty);
                // A settable bound property is stored in an `Option`, `None`
                // until set; anything else is stored directly.
                Some(match p.kind {
                    PropertyKind::Binding(_) => quote!(#field: Option<#ty>,),
                    PropertyKind::Stored(_) => quote!(#field: #ty,),
                })
            });
            let init = properties.iter().filter_map(|p| {
                let field = p.field.as_ref()?;
                Some(match &p.kind {
                    PropertyKind::Stored(init) => quote!(#field: #init,),
                    PropertyKind::Binding(_) => quote!(#field: None,),
                })
            });
            (quote!(pub struct #name { #(#fields)* }), quote!(Self { #(#init)* }))
        } else {
            (quote!(pub struct #name;), quote!(Self))
        };
        let accessors = properties.iter().map(|p| {
            let ty = &p.ty;
            let getter = p.getter.as_ref().map(|getter| {
                // A non-`Copy` field (a struct) is cloned on read.
                let read =
                    |field| if p.copy { quote!(self.#field) } else { quote!(self.#field.clone()) };
                let body = match &p.kind {
                    PropertyKind::Stored(_) => read(p.field.as_ref().unwrap()),
                    // A settable bound property falls back to the binding until
                    // set; a non-settable one always evaluates it.
                    PropertyKind::Binding(default) if p.setter.is_some() => {
                        let read = read(p.field.as_ref().unwrap());
                        quote!(#read.unwrap_or_else(|| #default))
                    }
                    PropertyKind::Binding(default) => quote!(#default),
                };
                quote!(pub fn #getter(&self) -> #ty { #body })
            });
            let setter = p.setter.as_ref().map(|setter| {
                let field = p.field.as_ref().expect("a setter belongs to a field");
                let assign = match p.kind {
                    PropertyKind::Binding(_) => quote!(self.#field = Some(value);),
                    PropertyKind::Stored(_) => quote!(self.#field = value;),
                };
                quote!(pub fn #setter(&mut self, value: #ty) { #assign })
            });
            quote!(#getter #setter)
        });
        output.extend(quote! {
            #struct_decl
            impl #name {
                pub fn new() -> Self {
                    #new_body
                }

                #(#accessors)*

                /// Render the window into a frame buffer of packed RGB triplets,
                /// whose length must be `width * height * 3`.
                pub fn render_rgb8(&self, width: u32, height: u32, frame_buffer: &mut [u8]) -> Result<(), slint_sc::RenderError> {
                    if frame_buffer.len() != width as usize * height as usize * 3 {
                        return Err(slint_sc::RenderError::InvalidFrameBufferSize);
                    }
                    let offset_x = 0i32;
                    let offset_y = 0i32;
                    #render_tree
                    Ok(())
                }
            }
        });
    }

    output.extend(images.statics());

    Ok(output)
}

struct DeclaredProperty {
    /// The value type `T`: `i32` for a length, `slint_sc::Color` for a color.
    ty: TokenStream,
    /// Whether `T` is `Copy`, so a stored field is read without cloning.
    copy: bool,
    /// The struct field `property_foo`.
    /// A non-settable bound property has none: nothing can set it, so the getter
    /// always evaluates the binding.
    field: Option<Ident>,
    kind: PropertyKind,
    /// `get_foo`, unless the property is private.
    getter: Option<Ident>,
    /// `set_foo`, for an `in` or `in-out` property.
    setter: Option<Ident>,
}

/// How a declared property is stored and read.
///
/// The binding, if any, lives in the getter (which has `self`), never in the
/// field's initial value, so `new()` never needs to evaluate it.
enum PropertyKind {
    /// No binding: a `T` field the getter reads and the setter writes, its
    /// initial value the type default. The token is that default.
    Stored(TokenStream),
    /// A binding. A settable property holds it in an `Option<T>` field, `None`
    /// until set, so the getter falls back to the binding and the property
    /// tracks it until set; a non-settable one evaluates the binding on read.
    /// The token is the compiled binding.
    Binding(TokenStream),
}

/// Whether a property with this visibility is settable (`in`/`in-out`).
fn is_settable(visibility: &PropertyVisibility) -> bool {
    matches!(visibility, PropertyVisibility::Input | PropertyVisibility::InOut)
}

/// The properties declared in the source on the component's root element.
/// Compiler-introduced declarations have no syntax node and are skipped.
fn declared_properties(ctx: &Ctx) -> Vec<DeclaredProperty> {
    let root_borrowed = ctx.root.borrow();
    root_borrowed
        .property_declarations
        .iter()
        .filter(|(_, decl)| decl.node.is_some())
        .map(|(name, decl)| {
            let snake = name.replace('-', "_");
            let ty = rust_type(&decl.property_type);
            let settable = is_settable(&decl.visibility);
            let has_getter = matches!(
                decl.visibility,
                PropertyVisibility::Input | PropertyVisibility::Output | PropertyVisibility::InOut
            );
            let kind = match root_borrowed.binding_cell_including_synthetic(name) {
                Some(b) => PropertyKind::Binding(compile_expression(&b.borrow().expression, ctx)),
                None => PropertyKind::Stored(default_value(&decl.property_type)),
            };
            // A non-settable binding is never stored, so it needs no field.
            let has_field = settable || matches!(kind, PropertyKind::Stored(_));
            DeclaredProperty {
                field: has_field.then(|| format_ident!("property_{snake}")),
                getter: has_getter.then(|| format_ident!("get_{snake}")),
                setter: settable.then(|| format_ident!("set_{snake}")),
                copy: is_copy(&decl.property_type),
                ty,
                kind,
            }
        })
        .collect()
}

/// The Rust type holding a value of the given Slint type.
fn rust_type(ty: &Type) -> TokenStream {
    match ty {
        Type::Int32 | Type::LogicalLength => quote!(i32),
        Type::Bool => quote!(bool),
        Type::Color => quote!(slint_sc::Color),
        Type::Image => quote!(slint_sc::Image),
        // A user-declared struct or enum maps to the generated type of the same name.
        Type::Struct(s) => {
            let StructName::User { name, .. } = &s.name else { unreachable!() };
            let name = format_ident!("{}", name.as_str());
            quote!(#name)
        }
        Type::Enumeration(en) => {
            let name = format_ident!("{}", en.name.as_str());
            quote!(#name)
        }
        // brush is not a declarable property type, and every other type was
        // rejected by the compiler
        _ => unreachable!(),
    }
}

/// The Rust value a property of the given type defaults to.
fn default_value(ty: &Type) -> TokenStream {
    match ty {
        Type::Int32 | Type::LogicalLength => quote!(0i32),
        Type::Bool => quote!(false),
        Type::Color => quote!(slint_sc::Color::default()),
        Type::Image => quote!(slint_sc::Image::None),
        Type::Struct(_) | Type::Enumeration(_) => {
            let ty = rust_type(ty);
            quote!(#ty::default())
        }
        _ => unreachable!(),
    }
}

/// Whether a value of the type is `Copy`, so a stored field can be read without
/// cloning. Only structs are not `Copy`.
fn is_copy(ty: &Type) -> bool {
    !matches!(ty, Type::Struct(_))
}

/// Emit the Rust definition of a user-declared struct or enum.
fn generate_type_definition(ty: &Type) -> TokenStream {
    match ty {
        Type::Struct(s) => {
            let StructName::User { name, field_order, .. } = &s.name else { unreachable!() };
            let name = format_ident!("{}", name.as_str());
            let fields = field_order.iter().map(|f| {
                let field = format_ident!("{}", f.replace('-', "_"));
                let ty = rust_type(&s.fields[f]);
                quote!(pub #field: #ty,)
            });
            quote! {
                #[derive(Default, Clone, PartialEq, Debug)]
                pub struct #name { #(#fields)* }
            }
        }
        Type::Enumeration(en) => {
            let name = format_ident!("{}", en.name.as_str());
            let variants = (0..en.values.len()).map(|value| {
                let variant = format_ident!(
                    "{}",
                    EnumerationValue { value, enumeration: en.clone() }.to_pascal_case()
                );
                if value == en.default_value {
                    quote!(#[default] #variant,)
                } else {
                    quote!(#variant,)
                }
            });
            quote! {
                #[derive(Default, Clone, Copy, PartialEq, Debug)]
                pub enum #name { #(#variants)* }
            }
        }
        // `structs_and_enums` holds only user structs and enums.
        _ => unreachable!(),
    }
}

/// Compile an expression of the Slint SC subset into Rust code. Lengths
/// become `i32` expressions and colors `slint_sc::Color` values. Besides the
/// literals of the subset, this handles the expressions that compiler passes
/// generate, like the centering arithmetic of default_geometry.
fn compile_expression(expr: &Expression, ctx: &Ctx) -> TokenStream {
    match expr {
        Expression::NumberLiteral(value, Unit::Px | Unit::None) => {
            let value = *value as i32;
            quote!(#value)
        }
        Expression::BoolLiteral(value) => quote!(#value),
        Expression::Cast { from, to: Type::Color | Type::Brush } => match from.as_ref() {
            Expression::NumberLiteral(value, _) => {
                let argb = *value as u32;
                quote!(slint_sc::Color::from_argb_encoded(#argb))
            }
            from => compile_expression(from, ctx),
        },
        // `int`, `float`, and `length` are all `i32` at runtime, so a cast
        // between them lowers to the operand itself.
        Expression::Cast { from, to: Type::Int32 | Type::Float32 | Type::LogicalLength } => {
            compile_expression(from, ctx)
        }
        Expression::PropertyReference(nr) => {
            // An unbound property has its type's default value.
            compile_property_reference(nr, ctx).unwrap_or_else(|| default_value(&nr.ty()))
        }
        Expression::ImageReference { resource_ref, .. } => match resource_ref {
            ImageReference::None => quote!(slint_sc::Image::None),
            ImageReference::EmbeddedTexture { resource_id } => {
                ctx.images.image_expression(*resource_id)
            }
            // Path, Url, and data-URI references were embedded into static
            // pixels, or rejected, by the compiler
            _ => unreachable!(),
        },
        Expression::EnumerationValue(ev) => {
            let enum_name = format_ident!("{}", ev.enumeration.name.as_str());
            let variant = format_ident!("{}", ev.to_pascal_case());
            quote!(#enum_name::#variant)
        }
        // A struct literal builds a value of a user struct; conversion has
        // already filled every field, so each field is emitted in order.
        Expression::Struct { ty, values } => {
            let StructName::User { name, field_order, .. } = &ty.name else { unreachable!() };
            let name = format_ident!("{}", name.as_str());
            let fields = field_order.iter().map(|f| {
                let field = format_ident!("{}", f.replace('-', "_"));
                let value = match values.get(f) {
                    Some(v) => compile_expression(v, ctx),
                    None => default_value(&ty.fields[f]),
                };
                quote!(#field: #value,)
            });
            quote!(#name { #(#fields)* })
        }
        // Field access reads a field out of a struct value.
        Expression::StructFieldAccess { base, name } => match base.as_ref() {
            // The dimensions of an image value, `some-image.width`: a field of
            // the ImageSize builtin call the lookup wraps around the image.
            // The dimensions are `usize` at runtime and `i32` in the language;
            // a dimension beyond `i32` saturates.
            Expression::FunctionCall {
                function: Callable::Builtin(BuiltinFunction::ImageSize),
                arguments,
                ..
            } => {
                let image = compile_expression(&arguments[0], ctx);
                let accessor = format_ident!("{}", name.as_str());
                quote!(i32::try_from((#image).#accessor()).unwrap_or(i32::MAX))
            }
            base => {
                let base = compile_expression(base, ctx);
                let field = format_ident!("{}", name.replace('-', "_"));
                quote!((#base).#field)
            }
        },
        Expression::BinaryExpression { lhs, rhs, op } => {
            let lhs = compile_expression(lhs, ctx);
            let rhs = compile_expression(rhs, ctx);
            // Arithmetic saturates at the `i32` bounds. `/` only comes from
            // compiler-synthesized geometry, whose divisor is a non-zero constant
            // (`saturating_div` still panics on a zero divisor).
            match op {
                '+' => quote!((#lhs).saturating_add(#rhs)),
                '-' => quote!((#lhs).saturating_sub(#rhs)),
                '*' => quote!((#lhs).saturating_mul(#rhs)),
                '/' => quote!((#lhs).saturating_div(#rhs)),
                // `&&` is `'&'` and `||` is `'|'`.
                '&' => quote!((#lhs) && (#rhs)),
                '|' => quote!((#lhs) || (#rhs)),
                // Comparison produces a `bool`. `==` is `'='` and `!=` is `'!'`;
                // `<=` is `'≤'` and `>=` is `'≥'`.
                '=' => quote!((#lhs) == (#rhs)),
                '!' => quote!((#lhs) != (#rhs)),
                '<' => quote!((#lhs) < (#rhs)),
                '>' => quote!((#lhs) > (#rhs)),
                '≤' => quote!((#lhs) <= (#rhs)),
                '≥' => quote!((#lhs) >= (#rhs)),
                _ => unreachable!(),
            }
        }
        Expression::UnaryOp { sub, op } => {
            let sub = compile_expression(sub, ctx);
            match op {
                // Negation saturates so that negating `i32::MIN` is defined.
                '-' => quote!((#sub).saturating_neg()),
                '+' => sub,
                '!' => quote!(!(#sub)),
                _ => unreachable!(),
            }
        }
        Expression::Condition { condition, true_expr, false_expr } => {
            let condition = compile_expression(condition, ctx);
            let true_expr = compile_expression(true_expr, ctx);
            let false_expr = compile_expression(false_expr, ctx);
            quote!(if #condition { #true_expr } else { #false_expr })
        }
        // A property read that appears more than once is hoisted into a local
        // variable, so a code block evaluates it once and reads it back.
        Expression::CodeBlock(statements) => {
            let statements = statements.iter().map(|s| compile_expression(s, ctx));
            quote!({ #(#statements)* })
        }
        Expression::StoreLocalVariable { name, value } => {
            let name = format_ident!("{}", name.replace('-', "_"));
            let value = compile_expression(value, ctx);
            quote!(let #name = #value;)
        }
        Expression::ReadLocalVariable { name, .. } => {
            let name = format_ident!("{}", name.replace('-', "_"));
            quote!(#name)
        }
        // Everything else was rejected by the compiler
        _ => unreachable!(),
    }
}

/// Whether the element is an image item: its resolved native class is, or
/// inherits, the class that declares `source`. Class selection may land on
/// the ImageItem base or a subclass of it, so the ancestry decides.
fn is_image_item(elem: &ElementRc) -> bool {
    let mut class = elem.borrow().native_class();
    while let Some(native) = class {
        if native.class_name == "ImageItem" {
            return true;
        }
        class = native.parent.clone();
    }
    false
}

/// Emit the render code for `elem` and its descendants: a block that adds the
/// element's position to the running `offset_x`/`offset_y`, paints its
/// background if it has one, and nests the children's blocks so that later
/// and deeper elements paint on top. An element without anything to paint in
/// its subtree emits nothing.
fn emit_element(elem: &ElementRc, ctx: &Ctx) -> TokenStream {
    let geometry = elem.borrow().geometry_props.clone();
    let resolve =
        |nr: Option<&NamedReference>| nr.and_then(|nr| compile_property_reference(nr, ctx));
    // The default_geometry pass leaves a size binding on every element, and
    // the root's size resolves to the window size
    let w = resolve(geometry.as_ref().map(|g| &g.width)).expect("element without a width");
    let h = resolve(geometry.as_ref().map(|g| &g.height)).expect("element without a height");
    let x = resolve(geometry.as_ref().map(|g| &g.x)).unwrap_or_else(|| quote!(0i32));
    let y = resolve(geometry.as_ref().map(|g| &g.y)).unwrap_or_else(|| quote!(0i32));
    let background = elem
        .borrow()
        .binding_cell_including_synthetic("background")
        .map(|b| b.borrow().expression.clone());
    let mut color = background.map(|expr| compile_expression(&expr, ctx));
    if Rc::ptr_eq(elem, ctx.root) {
        // The window background defaults to black, so that the whole frame
        // buffer is always painted
        color = Some(
            color.unwrap_or_else(|| quote!(slint_sc::Color::from_argb_encoded(0xff000000u32))),
        );
    }
    let fill = color.map(|color| {
        quote!(
            slint_sc::private_unstable_api::renderer::fill_rect(frame_buffer, [width, height],
                [offset_x, offset_y], [#w, #h], #color);
        )
    });
    // An image item paints its source image at the element position, one
    // image pixel per frame-buffer pixel: the element is always the size of
    // the image, so there is nothing to scale or clip to.
    let source = is_image_item(elem)
        .then(|| {
            elem.borrow()
                .binding_cell_including_synthetic("source")
                .map(|b| b.borrow().expression.clone())
        })
        .flatten();
    let draw_image = source.map(|source| {
        let source = compile_expression(&source, ctx);
        quote!(
            slint_sc::private_unstable_api::renderer::draw_image(frame_buffer,
                [width, height], [offset_x, offset_y], #source);
        )
    });
    let children: Vec<TokenStream> =
        elem.borrow().children.iter().map(|child| emit_element(child, ctx)).collect();
    if fill.is_none() && draw_image.is_none() && children.iter().all(|c| c.is_empty()) {
        return TokenStream::new();
    }
    quote! {
        {
            let offset_x = offset_x + #x;
            let offset_y = offset_y + #y;
            #fill
            #draw_image
            #(#children)*
        }
    }
}

/// Compile a reference to a property.
fn compile_property_reference(nr: &NamedReference, ctx: &Ctx) -> Option<TokenStream> {
    let element = nr.element();
    let is_root = Rc::ptr_eq(&element, ctx.root);
    if is_root {
        let root_borrowed = ctx.root.borrow();
        if let Some(decl) =
            root_borrowed.property_declarations.get(nr.name()).filter(|d| d.node.is_some())
        {
            let binding = root_borrowed.binding_cell_including_synthetic(nr.name());
            let snake = nr.name().replace('-', "_");
            return Some(match binding {
                None => {
                    let field = format_ident!("property_{snake}");
                    if is_copy(&decl.property_type) {
                        quote!(self.#field)
                    } else {
                        quote!(self.#field.clone())
                    }
                }
                Some(_) if is_settable(&decl.visibility) => {
                    let getter = format_ident!("get_{snake}");
                    quote!(self.#getter())
                }
                Some(b) => compile_expression(&b.borrow().expression, ctx),
            });
        }
    }
    match element.borrow().binding_cell_including_synthetic(nr.name()) {
        Some(b) => Some(compile_expression(&b.borrow().expression, ctx)),
        None if is_root => match nr.name().as_str() {
            "width" => Some(quote!((width as i32))),
            "height" => Some(quote!((height as i32))),
            _ => None,
        },
        None => None,
    }
}
