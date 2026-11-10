# العربية: نستورد أدوات كتابة وتشغيل اختبارات بايثون.
# English: Import Python's tools for defining and running tests.
import unittest

# العربية: نستورد PyTorch لإنشاء الموترات وتشغيل النموذج.
# English: Import PyTorch to create tensors and run the model.
import torch
# العربية: نستورد طبقات الشبكات العصبية، ومنها دالة حساب الخسارة.
# English: Import neural-network layers, including the loss function.
from torch import nn

# العربية: نستورد نموذج GPT الذي سنختبره.
# English: Import the GPT model that these tests exercise.
from tiny_gpt_model import TinyGPT


# العربية: نجمع اختبارات النموذج في فئة تتعرف إليها unittest.
# English: Group the model checks in a class recognized by unittest.
class TinyGPTModelTests(unittest.TestCase):
    # العربية: يتحقق الاختبار من أن الرموز المستقبلية لا تغيّر التوقعات السابقة.
    # English: Check that future tokens cannot change predictions at earlier positions.
    def test_future_tokens_do_not_change_earlier_logits(self):
        # العربية: نثبّت العشوائية لتسهيل تكرار نتيجة الاختبار.
        # English: Fix the random seed so the test is reproducible.
        torch.manual_seed(0)
        # العربية: ننشئ نموذجًا صغيرًا بإعدادات مناسبة للاختبار السريع.
        # English: Create a small model with settings suitable for a quick test.
        model = TinyGPT(
            # العربية: النموذج يتوقع واحدًا من 16 رمزًا ممكنًا.
            # English: The model predicts among 16 possible token IDs.
            vocab_size=16,
            # العربية: نمثل كل رمز بمتجه حجمه 16.
            # English: Represent each token with a 16-dimensional embedding.
            embedding_dim=16,
            # العربية: نستخدم أربعة رؤوس انتباه.
            # English: Use four attention heads.
            num_heads=4,
            # العربية: أقصى طول للتسلسل في هذا النموذج هو ثمانية رموز.
            # English: The model's maximum sequence length is eight tokens.
            block_size=8,
            # العربية: يتكون النموذج من طبقتي Transformer.
            # English: Build the model with two Transformer blocks.
            num_layers=2,
            # العربية: نضبط الإسقاط العشوائي على 20 بالمئة أثناء التدريب.
            # English: Set dropout to 20 percent during training.
            dropout=0.2,
        )
        # العربية: نفعّل وضع التقييم لتعطيل الإسقاط العشوائي.
        # English: Switch to evaluation mode to disable dropout.
        model.eval()

        # العربية: ننشئ التسلسل الأصلي المكوّن من ثمانية رموز.
        # English: Create the original sequence of eight token IDs.
        original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        # العربية: ننسخ التسلسل حتى لا نغيّر النسخة الأصلية.
        # English: Copy the sequence so the original remains unchanged.
        changed = original.clone()
        # العربية: نبدّل الرموز من الفهرس 4 إلى النهاية فقط.
        # English: Change only the tokens from index 4 to the end.
        changed[:, 4:] = torch.tensor([[10, 11, 12, 13]])

        # العربية: لا نحتاج إلى حساب التدرجات عند مقارنة التوقعات.
        # English: Gradients are unnecessary when comparing predictions.
        with torch.no_grad():
            # العربية: نحسب درجات النموذج لكل موضع في التسلسل الأصلي.
            # English: Compute the model's logits at every original position.
            original_logits = model(original)
            # العربية: نحسب الدرجات مرة أخرى بعد تغيير الرموز المستقبلية.
            # English: Recompute the logits after changing the future tokens.
            changed_logits = model(changed)

        # العربية: نتحقق من تطابق درجات أول أربعة مواضع في التسلسلين.
        # English: Verify that the logits at the first four positions match.
        torch.testing.assert_close(
            # العربية: درجات المواضع الأربعة الأولى من التسلسل الأصلي.
            # English: Logits for the first four original positions.
            original_logits[:, :4],
            # العربية: درجات المواضع الأربعة الأولى بعد تغيير المستقبل.
            # English: Logits for those positions after changing the future.
            changed_logits[:, :4],
        )

    # العربية: يتحقق الاختبار من قدرة النموذج على تعلّم دفعة صغيرة وحفظها.
    # English: Check that the model can learn and overfit one tiny batch.
    def test_model_can_overfit_a_tiny_batch(self):
        # العربية: نثبّت العشوائية للحصول على اختبار قابل للتكرار.
        # English: Fix the random seed to make the test reproducible.
        torch.manual_seed(0)
        # العربية: ننشئ نموذجًا صغيرًا بمفردات من ثمانية رموز.
        # English: Create a small model with an eight-token vocabulary.
        model = TinyGPT(
            # العربية: توجد ثمانية رموز مختلفة يمكن للنموذج توقعها.
            # English: The model can predict among eight token IDs.
            vocab_size=8,
            # العربية: حجم متجه تمثيل كل رمز هو 16.
            # English: Give each token a 16-dimensional embedding.
            embedding_dim=16,
            # العربية: نقسم الانتباه إلى أربعة رؤوس.
            # English: Divide attention into four heads.
            num_heads=4,
            # العربية: طول كل تسلسل في الدفعة هو ثمانية رموز.
            # English: Each sequence in the batch contains eight tokens.
            block_size=8,
            # العربية: نستخدم طبقتي Transformer.
            # English: Use two Transformer blocks.
            num_layers=2,
            # العربية: نعطّل الإسقاط العشوائي حتى يسهل حفظ الدفعة.
            # English: Disable dropout so the model can memorize the batch.
            dropout=0.0,
        )
        # العربية: ننشئ دفعة تدريب فيها أربعة تسلسلات قصيرة.
        # English: Create a training batch containing four short sequences.
        x = torch.tensor([
            # العربية: التسلسل الأول يضم الرموز من صفر إلى سبعة.
            # English: The first sequence contains token IDs zero through seven.
            [0, 1, 2, 3, 4, 5, 6, 7],
            # العربية: التسلسل الثاني هو تدوير للتسلسل الأول بمقدار خطوة.
            # English: The second sequence shifts the first sequence by one step.
            [1, 2, 3, 4, 5, 6, 7, 0],
            # العربية: التسلسل الثالث يدوّر الرموز بمقدار خطوتين.
            # English: The third sequence shifts the tokens by two steps.
            [2, 3, 4, 5, 6, 7, 0, 1],
            # العربية: التسلسل الرابع يدوّر الرموز بمقدار ثلاث خطوات.
            # English: The fourth sequence shifts the tokens by three steps.
            [3, 4, 5, 6, 7, 0, 1, 2],
        ])
        # العربية: نجعل الهدف هو الرمز التالي، مع إعادة أول رمز في نهاية التسلسل.
        # English: Use the next token as the target, wrapping the first token to the end.
        y = torch.roll(x, shifts=-1, dims=1)
        # العربية: نستخدم خسارة التصنيف المناسبة للتنبؤ برمز من عدة رموز.
        # English: Use the classification loss for choosing among token IDs.
        loss_fn = nn.CrossEntropyLoss()
        # العربية: نستخدم AdamW بمعدل تعلم قدره 0.01 لتحديث أوزان النموذج.
        # English: Use AdamW with a 0.01 learning rate to update model weights.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # العربية: نقيس الخسارة الابتدائية في وضع التقييم.
        # English: Measure the initial loss in evaluation mode.
        model.eval()
        # العربية: نعطّل حساب التدرجات أثناء قياس الخسارة الأولية.
        # English: Disable gradient tracking while measuring initial loss.
        with torch.no_grad():
            # العربية: نحسب الخسارة بين توقعات النموذج والأهداف الصحيحة.
            # English: Compute the loss between model predictions and targets.
            initial_loss = loss_fn(
                # العربية: نحوّل الدرجات إلى صفوف؛ كل صف يمثل توقع رمز واحد.
                # English: Flatten logits so each row predicts one token.
                model(x).reshape(-1, 8),
                # العربية: نحوّل الأهداف إلى قائمة الرموز الصحيحة المقابلة.
                # English: Flatten the matching list of correct token IDs.
                y.reshape(-1),
            # العربية: نحول قيمة الخسارة من موتر إلى رقم بايثون.
            # English: Convert the loss tensor into a regular Python number.
            ).item()

        # العربية: نفعّل وضع التدريب قبل تحديث أوزان النموذج.
        # English: Switch to training mode before updating model weights.
        model.train()
        # العربية: نكرر التدريب على الدفعة نفسها 200 مرة.
        # English: Train on the same batch for 200 optimization steps.
        for _ in range(200):
            # العربية: نحسب درجات التوقع لكل رمز في الدفعة.
            # English: Compute prediction logits for every token in the batch.
            logits = model(x)
            # العربية: نقيس الفرق بين التوقعات والأهداف الصحيحة.
            # English: Measure the difference between predictions and targets.
            loss = loss_fn(logits.reshape(-1, 8), y.reshape(-1))
            # العربية: نمسح التدرجات القديمة قبل حساب تدرجات هذه الخطوة.
            # English: Clear old gradients before computing gradients for this step.
            optimizer.zero_grad(set_to_none=True)
            # العربية: نحسب التدرجات التي توضّح كيف نخفّض الخسارة.
            # English: Compute gradients that indicate how to reduce the loss.
            loss.backward()
            # العربية: نحدّث أوزان النموذج باستخدام التدرجات.
            # English: Update the model weights using those gradients.
            optimizer.step()

        # العربية: نعيد النموذج إلى وضع التقييم لقياس النتيجة النهائية.
        # English: Return to evaluation mode to measure the final result.
        model.eval()
        # العربية: لا نحسب تدرجات أثناء قياس الخسارة النهائية.
        # English: Do not track gradients while measuring the final loss.
        with torch.no_grad():
            # العربية: نحسب خسارة النموذج بعد تدريبه على الدفعة.
            # English: Compute the loss after training on the batch.
            final_loss = loss_fn(
                # العربية: نحوّل توقعات النموذج إلى صفوف، صف لكل رمز.
                # English: Flatten model logits into one row per token.
                model(x).reshape(-1, 8),
                # العربية: نحوّل الرموز المستهدفة إلى الشكل نفسه.
                # English: Flatten the target tokens into the matching shape.
                y.reshape(-1),
            # العربية: نحوّل الخسارة النهائية إلى رقم بايثون.
            # English: Convert the final loss tensor into a Python number.
            ).item()

        # العربية: نشترط أن تصبح الخسارة أقل من عُشر قيمتها الأولية.
        # English: Require the final loss to fall below one tenth of its initial value.
        self.assertLess(final_loss, initial_loss * 0.1)


# العربية: شغّل الاختبارات عند تنفيذ هذا الملف مباشرةً.
# English: Run the tests when this file is executed directly.
if __name__ == "__main__":
    # العربية: ابدأ مشغّل اختبارات unittest.
    # English: Start the unittest test runner.
    unittest.main()
