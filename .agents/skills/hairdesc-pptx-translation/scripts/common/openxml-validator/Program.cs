using System.Text.Json;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;

if (args.Length != 1) {
    Console.Error.WriteLine("Usage: Validator presentation.pptx");
    return 2;
}
try {
    using var doc = PresentationDocument.Open(args[0], false);
    var validator = new OpenXmlValidator(FileFormatVersions.Microsoft365) { MaxNumberOfErrors = 0 };
    var errors = validator.Validate(doc).Select(e => new {
        id = e.Id, type = e.ErrorType.ToString(), part = e.Part?.Uri.ToString(),
        path = e.Path?.XPath, related = e.RelatedPart?.Uri.ToString(), description = e.Description
    }).ToArray();
    Console.WriteLine(JsonSerializer.Serialize(new {
        target = "Microsoft365", sdk = typeof(OpenXmlValidator).Assembly.GetName().Version?.ToString(), errors
    }));
    return 0;  // A complete diagnostic report; the caller compares the two multisets.
}
catch (Exception e) {
    Console.Error.WriteLine(e.Message);
    return 2;
}
