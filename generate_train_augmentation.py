"""LLM-authored templates, deterministically combined by Python; TRAIN ONLY.

No original review, blind text, star rating or external labeling API is used.
The generated CSV and provenance must accompany any submission using it.
"""
from pathlib import Path
import json,random,hashlib
import pandas as pd
from absa_core import ASPECTS,LABELS

ROOT=Path(__file__).resolve().parent
PHRASES={
'nhan_vien':{
'khen':['Nhân viên hỗ trợ rất nhiệt tình.','Lễ tân giải quyết yêu cầu nhanh chóng.','Thái độ phục vụ thân thiện và lịch sự.','Nhận phòng nhanh, nhân viên hướng dẫn rõ ràng.','Nhân viên xử lý sự cố rất chu đáo.','Lễ tân chủ động giúp đỡ khi mình cần.','Nhân viên luôn niềm nở với khách.','Trả phòng nhanh, phục vụ chuyên nghiệp.'],
'che':['Lễ tân làm thủ tục quá chậm.','Nhân viên trả lời thiếu lịch sự.','Thái độ phục vụ chỉ tạm được.','Nhân viên chưa nhiệt tình hỗ trợ khách.','Chờ mãi mà không có ai xử lý yêu cầu.','Dịch vụ còn cần cải thiện nhiều.','Lễ tân không giải thích rõ khi khách hỏi.','Nhân viên phục vụ bình thường, chưa chu đáo.']},
'sach_se':{
'khen':['Ga gối sạch sẽ và không có mùi.','Nhà tắm được dọn rất sạch.','Không gian sạch, không thấy bụi.','Chăn gối thơm tho.','Sàn được lau sạch mỗi ngày.','Khu vệ sinh sạch sẽ.','Phòng luôn được giữ sạch.','Đồ vải rất sạch và thơm.'],
'che':['Ga gối có vết bẩn.','Nhà tắm còn mùi hôi.','Chăn có mùi ẩm mốc.','Vệ sinh chỉ tạm được, cần dọn kỹ hơn.','Sàn vẫn còn bụi.','Trong phòng có gián và muỗi.','Chăn gối chưa được giặt sạch.','Độ sạch sẽ bình thường, chưa tốt.']},
'vi_tri':{
'khen':['Vị trí gần biển, đi bộ rất tiện.','Khách sạn nằm ngay trung tâm.','Đi lại đến các điểm tham quan thuận lợi.','Xung quanh có nhiều quán ăn tiện lợi.','Địa điểm dễ tìm và dễ di chuyển.','Khu vực xung quanh yên tĩnh.','Từ khách sạn đi ra biển rất gần.','Vị trí thuận tiện cho chuyến đi.'],
'che':['Vị trí khá xa trung tâm.','Đường đến khách sạn khó tìm.','Đi ra biển mất nhiều thời gian.','Khu vực bên ngoài quá ồn vào buổi tối.','Xung quanh ít hàng quán nên đi lại bất tiện.','Đường vào khách sạn đang sửa nên khó di chuyển.','Vị trí chỉ tạm được, không thuận tiện.','Tiếng xe ngoài đường gây ồn suốt đêm.']},
'phong':{
'khen':['Phòng rộng và bố trí hợp lý.','Giường nằm êm và dễ ngủ.','Điều hòa hoạt động tốt.','Cách âm tốt, ngủ rất ngon.','Ban công nhìn ra biển rất đẹp.','Nội thất phòng mới và đẹp.','Phòng tắm rộng, nước nóng ổn định.','Phòng đầy đủ thiết bị cần thiết.'],
'che':['Phòng hơi chật.','Giường quá cứng nên khó ngủ.','Điều hòa không mát.','Cách âm kém, nghe rõ phòng bên cạnh.','Nội thất đã xuống cấp.','Nước nóng lúc có lúc không.','Tầm nhìn từ phòng bị che khuất.','Phòng chỉ tạm được, thiết bị cần nâng cấp.']},
'gia_tri':{
'khen':['Mức giá hợp lý so với những gì nhận được.','Chất lượng xứng đáng với giá tiền.','Giá rất phải chăng.','Ở đây đáng tiền.','Chi phí phù hợp với ngân sách.','Mức giá tốt cho trải nghiệm này.','Số tiền bỏ ra rất xứng đáng.','Giá cả ổn và không có phụ thu.'],
'che':['Giá quá cao so với chất lượng.','Không đáng số tiền đã trả.','Phụ thu khá nhiều.','Chi phí phát sinh không hợp lý.','Mức giá chỉ tạm chấp nhận được.','Giá hơi đắt cho trải nghiệm này.','Chất lượng chưa tương xứng với giá.','Cảm thấy không đáng tiền.']},
'an_uong':{
'khen':['Bữa sáng ngon và đa dạng.','Buffet sáng có nhiều món hấp dẫn.','Đồ ăn tại nhà hàng nêm nếm vừa miệng.','Cà phê của khách sạn rất ngon.','Thực đơn phong phú.','Các món ăn được chuẩn bị nóng và ngon.','Đồ uống ở quầy bar rất ổn.','Ăn sáng hợp khẩu vị.'],
'che':['Bữa sáng chỉ bình thường.','Buffet sáng ít món và hay bị hết.','Đồ ăn không ngon.','Thực đơn cần đa dạng hơn.','Cà phê của khách sạn hơi nhạt.','Ăn sáng tạm được, chưa hấp dẫn.','Các món ăn nguội và đơn điệu.','Đồ uống ở quầy bar quá ngọt.']},
'tien_nghi':{
'khen':['Wifi nhanh và ổn định.','Hồ bơi rộng và đẹp.','Phòng gym có nhiều máy tập.','Thang máy hoạt động nhanh.','Bãi đỗ xe rộng rãi.','Khuôn viên có nhiều cây xanh đẹp.','Dịch vụ đưa đón của khách sạn thuận tiện.','Khu vui chơi có nhiều trò cho trẻ nhỏ.'],
'che':['Wifi yếu và thường xuyên mất kết nối.','Hồ bơi hơi nhỏ.','Phòng gym có ít máy tập.','Thang máy chậm, phải chờ lâu.','Bãi đỗ xe chật và bất tiện.','Khuôn viên chỉ tạm được.','Khu vui chơi cần thêm nhiều trò hơn.','Dịch vụ đưa đón của khách sạn chưa đúng giờ.']}}

def generate(n=640):
    rng=random.Random(2026);rows=[];seen=set()
    while len(rows)<n:
        aspects=rng.sample(ASPECTS,rng.choice([1,2,2,3]))
        sentences=[];label={c:0 for c in LABELS}
        for a in aspects:
            polarity=rng.choices(['khen','che','both'],[.25,.55,.20])[0]
            pols=['khen','che'] if polarity=='both' else [polarity]
            fragments=[rng.choice(PHRASES[a][p]) for p in pols]
            sentences.append(' Tuy nhiên, '.join(fragments))
            label[a+'_nhac']=1
            for p in pols:label[a+'_'+p]=1
        text=' '.join(sentences)
        if text in seen:continue
        seen.add(text);rows.append({'id':f'aug_{len(rows)+1:05d}','text':text,**label})
    return pd.DataFrame(rows,columns=['id','text']+LABELS)

if __name__=='__main__':
    df=generate();target=ROOT/'du_lieu_them.csv';df.to_csv(target,index=False,encoding='utf-8')
    provenance={'file_to_include_in_final_zip':'du_lieu_them.csv','local_file':target.name,'rows':len(df),'seed':2026,
        'tool':'ChatGPT/Codex authored templates and labels; Python deterministic combinations; no external labeling API',
        'user_prompt':'do k được dùng file test_blind á nên chạy làm sao cho được 0.96 đi bạn tôi tiền xử lý dữ kiệu, phobert tfidf mà được trên 0.9 đó. tìm cách đi',
        'construction_spec':'Create novel Vietnamese review clauses for the assignment seven aspects; label mention/praise/complaint including mild complaint, temporary acceptance and improvement requests. Combine clauses deterministically. Never use blind text or original review retrieval.',
        'labels':'Manually specified by the assistant for synthetic templates; unions when combining clauses',
        'source_texts':'New templates in generate_train_augmentation.py, not copied class reviews',
        'blind_file_read':False,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (ROOT/'augmentation_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({'rows':len(df),'sha256':provenance['sha256'],'blind_file_read':False}))
